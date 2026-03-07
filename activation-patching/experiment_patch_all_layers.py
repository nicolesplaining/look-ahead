import json
import os
import torch
import pronouncing
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ─────────────────────────────────────────────────────────────────────

RUN_NAME = "Qwen3-32B-N100-T07-Pos3-all-layers"

MODEL_NAME = "Qwen/Qwen3-32B"

CLEAN_PROMPT   = "A rhyming couplet:\nShe felt a sudden sense of fright,\nand hoped that"
CORRUPT_PROMPT = "A rhyming couplet:\nShe felt a sudden sense of fear,\nand hoped that"

CLEAN_RHYME_WORD   = "fright"
CORRUPT_RHYME_WORD = "fear"

SAMPLING_N    = 100
SAMPLING_TEMP = 0.7
MAX_NEW_TOKENS = 12

# Position to patch (relative to the second newline token, which is i=0).
# i=-1 → one token before the second newline (e.g. the rhyme word / punctuation)
# i=-2 → two tokens before, etc.
# Note: Qwen may tokenize e.g. ",\n" as a single token; token boundaries are
# resolved via the tokenizer's offset mapping, so this is always exact.
PATCH_POSITION = 3

# ── Rhyme Checking ─────────────────────────────────────────────────────────────

def _rhyme_score(w1: str, w2: str) -> Optional[bool]:
    p1 = pronouncing.phones_for_word(w1.lower().strip())
    p2 = pronouncing.phones_for_word(w2.lower().strip())
    if not p1 or not p2:
        return None
    rp1 = pronouncing.rhyming_part(p1[0])
    rp2 = pronouncing.rhyming_part(p2[0])
    return (rp1 == rp2) if (rp1 and rp2) else None

# ── Rhyme Word Extraction ──────────────────────────────────────────────────────

def last_word(text: str) -> str:
    for w in reversed(text.split()):
        cleaned = w.strip(".,!?\"'—;: ")
        if cleaned.isalpha():
            return cleaned.lower()
    return ""

def word_before_nth_newline(text: str, n: int) -> str:
    if n <= 0:
        return ""
    newline_positions = [i for i, ch in enumerate(text) if ch == "\n"]
    if len(newline_positions) < n:
        return ""
    end   = newline_positions[n - 1]
    start = newline_positions[n - 2] + 1 if n >= 2 else 0
    return last_word(text[start:end])

def extract_rhyme_word(full_text: str, prompt: str) -> str:
    target_newline_index = prompt.count("\n") + 1
    rhyme_word = word_before_nth_newline(full_text, target_newline_index)
    if rhyme_word:
        return rhyme_word
    if full_text.startswith(prompt):
        return last_word(full_text[len(prompt):])
    return last_word(full_text)

def rhyme_rate(completions: list[str], prompt: str, rhyme_word: str) -> float:
    hits = sum(
        1 for c in completions
        if _rhyme_score(extract_rhyme_word(c, prompt), rhyme_word) is True
    )
    return hits / len(completions) if completions else 0.0

# ── Model Loading ───────────────────────────────────────────────────────────────

def load_model():
    print(f"Loading {MODEL_NAME} via transformers...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Loaded. Layers: {model.config.num_hidden_layers} | d_model: {model.config.hidden_size}")
    return model, tokenizer

# ── Generation Helpers ──────────────────────────────────────────────────────────

def get_input_device(model) -> torch.device:
    return model.model.embed_tokens.weight.device

def generate_text(model, tokenizer, prompt: str, temperature: float) -> str:
    device = get_input_device(model)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def sample_completions(model, tokenizer, prompt: str, n: int, temperature: float) -> list[str]:
    return [generate_text(model, tokenizer, prompt, temperature) for _ in range(n)]

# ── Activation Caching ──────────────────────────────────────────────────────────

def cache_hidden_states(model, tokenizer, prompt: str) -> tuple:
    """
    Returns tuple of (n_layers + 1) tensors [1, seq_len, d_model]:
      hidden_states[L] = resid_pre for layer L
    """
    device = get_input_device(model)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**enc, output_hidden_states=True)
    return tuple(h.detach() for h in outputs.hidden_states)

# ── Position Finding ────────────────────────────────────────────────────────────

def resolve_patch_pos(prompt: str, tokenizer, position_offset: int) -> tuple[int, str]:
    """
    Find the token index for `position_offset` relative to the second newline (i=0).
    Uses the tokenizer's offset mapping so fused tokens (e.g. Qwen's ",\\n") are
    handled correctly — we look for which token *covers* the second newline char.
    Returns (abs_token_pos, token_string).
    """
    newline_chars = [i for i, ch in enumerate(prompt) if ch == "\n"]
    if len(newline_chars) < 2:
        raise ValueError(f"Prompt must contain at least 2 newlines, found {len(newline_chars)}.")
    second_nl_char = newline_chars[1]

    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    offset_mapping = enc["offset_mapping"]

    second_nl_tok = next(
        (i for i, (s, e) in enumerate(offset_mapping) if s <= second_nl_char < e),
        None,
    )
    if second_nl_tok is None:
        raise ValueError("Could not find token covering the second newline in prompt.")

    abs_pos = second_nl_tok + position_offset
    n_tokens = len(offset_mapping)
    if abs_pos < 0 or abs_pos >= n_tokens:
        raise ValueError(
            f"PATCH_POSITION={position_offset} maps to abs_pos={abs_pos}, "
            f"which is out of bounds (prompt has {n_tokens} tokens)."
        )

    token_ids = enc["input_ids"]
    tok_str = tokenizer.decode([token_ids[abs_pos]])
    return abs_pos, tok_str, second_nl_tok

# ── Hook ────────────────────────────────────────────────────────────────────────

def make_patch_hook(patch_vec: torch.Tensor, patch_pos: int):
    """
    forward_pre_hook: replaces hidden_states[:, patch_pos, :] with patch_vec at prefill.
    No-op during decode steps (seq_len == 1).
    """
    def hook_fn(module, args):
        h = args[0]
        if h.shape[1] <= 1:
            return args
        if h.shape[1] > patch_pos:
            out = h.clone()
            out[:, patch_pos, :] = patch_vec.to(h.device)
            return (out,) + args[1:]
        return args
    return hook_fn

# ── Main Experiment ─────────────────────────────────────────────────────────────

def run_experiment():
    model, tokenizer = load_model()
    n_layers = model.config.num_hidden_layers

    # --- Resolve patch position via tokenizer offset mapping ---
    abs_patch_pos, patch_token, second_nl_tok = resolve_patch_pos(
        CLEAN_PROMPT, tokenizer, PATCH_POSITION
    )
    patch_label = f"i={PATCH_POSITION:+d} (abs={abs_patch_pos}, tok={repr(patch_token)})"

    clean_ids   = tokenizer(CLEAN_PROMPT,   return_tensors="pt").input_ids
    corrupt_ids = tokenizer(CORRUPT_PROMPT, return_tensors="pt").input_ids
    clean_seq_len   = clean_ids.shape[1]
    corrupt_seq_len = corrupt_ids.shape[1]
    if clean_seq_len != corrupt_seq_len:
        print(f"\nWARNING: token length mismatch ({clean_seq_len} vs {corrupt_seq_len})")

    print(f"\nSecond newline token: abs_pos={second_nl_tok} (i=0)")
    print(f"Patch position: {patch_label}")
    print(f"Patch direction: corrupt → clean, ALL {n_layers} layers simultaneously")
    print(f"N={SAMPLING_N}, T={SAMPLING_TEMP}")

    print(f"\nClean prompt tokens:")
    for idx, tok_id in enumerate(clean_ids[0].tolist()):
        offset = idx - second_nl_tok
        marker = " ← patch target" if idx == abs_patch_pos else ""
        if idx == second_nl_tok and idx != abs_patch_pos:
            marker = " ← i=0 (2nd newline)"
        print(f"  abs={idx:2d}  i={offset:+3d}  {repr(tokenizer.decode([tok_id]))}{marker}")

    # --- Greedy baselines ---
    print("\n── Baseline Completions (greedy) ──")
    clean_completion   = generate_text(model, tokenizer, CLEAN_PROMPT,   temperature=0)
    corrupt_completion = generate_text(model, tokenizer, CORRUPT_PROMPT, temperature=0)
    print(f"Clean   -> {repr(clean_completion)}")
    print(f"Corrupt -> {repr(corrupt_completion)}")
    clean_end   = extract_rhyme_word(clean_completion,   CLEAN_PROMPT)
    corrupt_end = extract_rhyme_word(corrupt_completion, CORRUPT_PROMPT)
    print(f"\nClean ends with:   '{clean_end}' — rhymes with '{CLEAN_RHYME_WORD}'?   {_rhyme_score(clean_end,   CLEAN_RHYME_WORD)}")
    print(f"Corrupt ends with: '{corrupt_end}' — rhymes with '{CORRUPT_RHYME_WORD}'? {_rhyme_score(corrupt_end, CORRUPT_RHYME_WORD)}")

    # ── Group 1: Baseline (unpatched clean run) ────────────────────────────────
    print(f"\n── Group 1: Unpatched Clean Baseline ({SAMPLING_N} samples, T={SAMPLING_TEMP}) ──")
    baseline_completions  = sample_completions(model, tokenizer, CLEAN_PROMPT, SAMPLING_N, SAMPLING_TEMP)
    baseline_clean_rate   = rhyme_rate(baseline_completions, CLEAN_PROMPT, CLEAN_RHYME_WORD)
    baseline_corrupt_rate = rhyme_rate(baseline_completions, CLEAN_PROMPT, CORRUPT_RHYME_WORD)
    print(f"  Rhymes with '{CLEAN_RHYME_WORD}'   (expected high): {baseline_clean_rate:.3f}")
    print(f"  Rhymes with '{CORRUPT_RHYME_WORD}' (expected low):  {baseline_corrupt_rate:.3f}")

    # --- Cache corrupt activations ---
    print("\nCaching corrupt activations...")
    corrupt_hs = cache_hidden_states(model, tokenizer, CORRUPT_PROMPT)

    # ── Group 2: All layers patched simultaneously at PATCH_POSITION ───────────
    print(f"\n── Group 2: All {n_layers} layers patched at {patch_label} ({SAMPLING_N} samples, T={SAMPLING_TEMP}) ──")
    handles = []
    for layer in range(n_layers):
        patch_vec = corrupt_hs[layer][:, abs_patch_pos, :].clone()
        handle = model.model.layers[layer].register_forward_pre_hook(
            make_patch_hook(patch_vec, abs_patch_pos)
        )
        handles.append(handle)

    try:
        patched_completions = sample_completions(model, tokenizer, CLEAN_PROMPT, SAMPLING_N, SAMPLING_TEMP)
        patched_clean_rate   = rhyme_rate(patched_completions, CLEAN_PROMPT, CLEAN_RHYME_WORD)
        patched_corrupt_rate = rhyme_rate(patched_completions, CLEAN_PROMPT, CORRUPT_RHYME_WORD)
    finally:
        for h in handles:
            h.remove()

    delta = patched_corrupt_rate - baseline_corrupt_rate
    print(f"  Rhymes with '{CLEAN_RHYME_WORD}'   (expect drop):  {patched_clean_rate:.3f}  (baseline: {baseline_clean_rate:.3f})")
    print(f"  Rhymes with '{CORRUPT_RHYME_WORD}' (expect rise):  {patched_corrupt_rate:.3f}  (baseline: {baseline_corrupt_rate:.3f}, delta: {delta:+.3f})")

    # --- Save JSON ---
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    run_dir     = os.path.join(results_dir, RUN_NAME)
    os.makedirs(run_dir, exist_ok=True)

    json_path = os.path.join(run_dir, "generations.json")
    with open(json_path, "w") as f:
        json.dump({
            "run_name":           RUN_NAME,
            "model_name":         MODEL_NAME,
            "patch_direction":    "corrupt→clean",
            "patch_mode":         "single_position_all_layers_simultaneous",
            "patch_position":     PATCH_POSITION,
            "second_nl_tok":      second_nl_tok,
            "abs_patch_pos":      abs_patch_pos,
            "patch_token":        patch_token,
            "clean_seq_len":      clean_seq_len,
            "corrupt_seq_len":    corrupt_seq_len,
            "n_layers":           n_layers,
            "sampling_n":         SAMPLING_N,
            "sampling_temp":      SAMPLING_TEMP,
            "max_new_tokens":     MAX_NEW_TOKENS,
            "clean_prompt":       CLEAN_PROMPT,
            "corrupt_prompt":     CORRUPT_PROMPT,
            "clean_rhyme_word":   CLEAN_RHYME_WORD,
            "corrupt_rhyme_word": CORRUPT_RHYME_WORD,
            "baseline": {
                "clean_completion":                   clean_completion,
                "corrupt_completion":                 corrupt_completion,
                "completions":                        baseline_completions,
                "clean_rhyme_rate":                   baseline_clean_rate,
                "corrupt_rhyme_rate":                 baseline_corrupt_rate,
            },
            "patched": {
                "completions":       patched_completions,
                "clean_rhyme_rate":  patched_clean_rate,
                "corrupt_rhyme_rate": patched_corrupt_rate,
                "delta_corrupt_rate": delta,
            },
        }, f, indent=2)
    print(f"\nGenerations saved to {json_path}")


if __name__ == "__main__":
    run_experiment()
