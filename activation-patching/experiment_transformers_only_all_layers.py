import json
import os
import torch
import pronouncing
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────

RUN_NAME = "qwen3-32b-N500-T07-newpoem-all-positions"

MODEL_NAME = "Qwen/Qwen3-32B"

# Patch direction: corrupt → clean (inject "fright" activations into "fear" run)
CLEAN_PROMPT   = "A rhyming couplet:\nShe felt a sudden sense of fear,\n"
CORRUPT_PROMPT = "A rhyming couplet:\nShe felt a sudden sense of fright,\n"

CLEAN_RHYME_WORD   = "fear"
CORRUPT_RHYME_WORD = "fright"

SAMPLING_N    = 500
SAMPLING_TEMP = 0.7
MAX_NEW_TOKENS = 12

# ── Rhyme Checking ─────────────────────────────────────────────────────────────

def _rhyme_score(w1: str, w2: str) -> Optional[bool]:
    """True if w1 and w2 rhyme, False if they don't, None if either is unknown."""
    p1 = pronouncing.phones_for_word(w1.lower().strip())
    p2 = pronouncing.phones_for_word(w2.lower().strip())
    if not p1 or not p2:
        return None
    rp1 = pronouncing.rhyming_part(p1[0])
    rp2 = pronouncing.rhyming_part(p2[0])
    return (rp1 == rp2) if (rp1 and rp2) else None

# ── Rhyme Word Extraction ──────────────────────────────────────────────────────

def last_word(text: str) -> str:
    """Last alphabetic word from text, scanning from the end."""
    for w in reversed(text.split()):
        cleaned = w.strip(".,!?\"'—;: ")
        if cleaned.isalpha():
            return cleaned.lower()
    return ""

def word_before_nth_newline(text: str, n: int) -> str:
    """
    Last word of the n-th line segment (the segment that ends at the n-th newline).
    Only scans that specific line — does not fall back to earlier lines.
    """
    if n <= 0:
        return ""
    newline_positions = [i for i, ch in enumerate(text) if ch == "\n"]
    if len(newline_positions) < n:
        return ""
    end   = newline_positions[n - 1]
    start = newline_positions[n - 2] + 1 if n >= 2 else 0
    return last_word(text[start:end])

def extract_rhyme_word(full_text: str, prompt: str) -> str:
    """
    Extract the rhyme word from the first generated line.
    Looks for the last word of the line ending at (prompt_newlines + 1)-th newline.
    Fallback: last word of the generated text if no such newline exists.
    """
    target_newline_index = prompt.count("\n") + 1
    rhyme_word = word_before_nth_newline(full_text, target_newline_index)
    if rhyme_word:
        return rhyme_word
    if full_text.startswith(prompt):
        return last_word(full_text[len(prompt):])
    return last_word(full_text)

def rhyme_rate(completions: list[str], prompt: str, rhyme_word: str) -> float:
    """Fraction of completions whose extracted rhyme word rhymes with rhyme_word."""
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
    """Generate text from prompt; returns full string (prompt + generation)."""
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
    """Draw n sampled completions from the model given a prompt."""
    return [generate_text(model, tokenizer, prompt, temperature) for _ in range(n)]

# ── Activation Caching & Patching ──────────────────────────────────────────────

def cache_hidden_states(model, tokenizer, prompt: str) -> tuple:
    """
    Run a forward pass with output_hidden_states=True.
    Returns tuple of (n_layers + 1) tensors [1, seq_len, d_model]:
      hidden_states[L] = resid_pre for layer L
    """
    device = get_input_device(model)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**enc, output_hidden_states=True)
    return tuple(h.detach() for h in outputs.hidden_states)

def make_patch_hook_all_positions(patch_tensor: torch.Tensor):
    """
    forward_pre_hook for model.model.layers[L].
    At prefill: replaces all token positions with the corresponding corrupt
    hidden states. If prompt lengths differ, patches up to the shorter length.
    No-op during decode steps (seq_len == 1).
    """
    corrupt_seq_len = patch_tensor.shape[1]
    def hook_fn(module, args):
        h = args[0]
        if h.shape[1] <= 1:  # decode step — skip
            return args
        n = min(h.shape[1], corrupt_seq_len)
        out = h.clone()
        out[:, :n, :] = patch_tensor[:, :n, :].to(h.device)
        return (out,) + args[1:]
    return hook_fn

# ── Main Experiment ─────────────────────────────────────────────────────────────

def run_experiment():
    model, tokenizer = load_model()
    n_layers = model.config.num_hidden_layers

    # --- Tokenize & check lengths ---
    clean_ids   = tokenizer(CLEAN_PROMPT,   return_tensors="pt").input_ids
    corrupt_ids = tokenizer(CORRUPT_PROMPT, return_tensors="pt").input_ids
    clean_seq_len   = clean_ids.shape[1]
    corrupt_seq_len = corrupt_ids.shape[1]
    if clean_seq_len != corrupt_seq_len:
        print(f"\nWARNING: token length mismatch ({clean_seq_len} vs {corrupt_seq_len})")
        print("Patching up to the shorter sequence length.")
    patch_positions = min(clean_seq_len, corrupt_seq_len)

    print(f"\nPatch direction: corrupt → clean (all {patch_positions} token positions)")
    print(f"  Injecting ALL '{CORRUPT_RHYME_WORD}' hidden states into '{CLEAN_RHYME_WORD}' run")
    print(f"  N={SAMPLING_N}, T={SAMPLING_TEMP}")

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

    # --- Sampling baseline: unpatched clean run ---
    print(f"\n── Unpatched Clean Baseline ({SAMPLING_N} samples, T={SAMPLING_TEMP}) ──")
    baseline_samples      = sample_completions(model, tokenizer, CLEAN_PROMPT, SAMPLING_N, SAMPLING_TEMP)
    baseline_clean_rate   = rhyme_rate(baseline_samples, CLEAN_PROMPT, CLEAN_RHYME_WORD)
    baseline_corrupt_rate = rhyme_rate(baseline_samples, CLEAN_PROMPT, CORRUPT_RHYME_WORD)
    print(f"  Rhymes with '{CLEAN_RHYME_WORD}' (expected high): {baseline_clean_rate:.3f}")
    print(f"  Rhymes with '{CORRUPT_RHYME_WORD}' (expected low): {baseline_corrupt_rate:.3f}")

    # --- Cache corrupt activations (source of patch) ---
    print("\nCaching corrupt activations...")
    corrupt_hs = cache_hidden_states(model, tokenizer, CORRUPT_PROMPT)

    # --- Layer sweep: run CLEAN prompt with ALL corrupt positions patched in ---
    print(f"\nPatching ALL positions across all {n_layers} layers (corrupt→clean, N={SAMPLING_N}, T={SAMPLING_TEMP})...\n")

    results = []

    for layer in tqdm(range(n_layers), desc="Layers"):
        patch_tensor = corrupt_hs[layer].clone()  # [1, corrupt_seq_len, d_model]
        handle = model.model.layers[layer].register_forward_pre_hook(
            make_patch_hook_all_positions(patch_tensor)
        )
        try:
            completions  = sample_completions(model, tokenizer, CLEAN_PROMPT, SAMPLING_N, SAMPLING_TEMP)
            clean_rate   = rhyme_rate(completions, CLEAN_PROMPT, CLEAN_RHYME_WORD)
            corrupt_rate = rhyme_rate(completions, CLEAN_PROMPT, CORRUPT_RHYME_WORD)
        finally:
            handle.remove()

        delta = corrupt_rate - baseline_corrupt_rate
        print(f"  Layer {layer:2d}: corrupt_rhyme_rate={corrupt_rate:.3f} (baseline={baseline_corrupt_rate:.3f}, delta={delta:+.3f})")
        results.append({
            "layer": layer,
            "completions": completions,
            "clean_rhyme_rate":      clean_rate,
            "corrupt_rhyme_rate":    corrupt_rate,
            "baseline_clean_rate":   baseline_clean_rate,
            "baseline_corrupt_rate": baseline_corrupt_rate,
        })

    # --- Summary ---
    print(f"\n── Summary ──")
    best = max(results, key=lambda r: r["corrupt_rhyme_rate"])
    print(f"Best layer: {best['layer']} (corrupt_rhyme_rate={best['corrupt_rhyme_rate']:.3f}, baseline={baseline_corrupt_rate:.3f})")

    # --- Save JSON ---
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    run_dir     = os.path.join(results_dir, RUN_NAME)
    os.makedirs(run_dir, exist_ok=True)

    json_path = os.path.join(run_dir, "generations.json")
    with open(json_path, "w") as f:
        json.dump({
            "run_name":    RUN_NAME,
            "model_name":  MODEL_NAME,
            "patch_direction": "corrupt→clean",
            "patch_mode":  "all_positions",
            "patch_positions": patch_positions,
            "clean_seq_len":   clean_seq_len,
            "corrupt_seq_len": corrupt_seq_len,
            "sampling_n":    SAMPLING_N,
            "sampling_temp": SAMPLING_TEMP,
            "max_new_tokens": MAX_NEW_TOKENS,
            "clean_prompt":   CLEAN_PROMPT,
            "corrupt_prompt": CORRUPT_PROMPT,
            "clean_rhyme_word":   CLEAN_RHYME_WORD,
            "corrupt_rhyme_word": CORRUPT_RHYME_WORD,
            "baseline": {
                "clean_completion":   clean_completion,
                "corrupt_completion": corrupt_completion,
                "unpatched_clean_clean_rhyme_rate":   baseline_clean_rate,
                "unpatched_clean_corrupt_rhyme_rate": baseline_corrupt_rate,
                "completions": baseline_samples,
            },
            "n_layers": n_layers,
            "results":  results,
        }, f, indent=2)
    print(f"Generations saved to {json_path}")

    return results


if __name__ == "__main__":
    results = run_experiment()
