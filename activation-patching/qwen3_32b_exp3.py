"""
Activation patching on Qwen3-32B — positions i=-1 and i=-2 relative to newline.

Patch direction: corrupt → clean
  i=-1: token ' fear'   (the rhyme word itself)
  i=-2: token ' of'
"""

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone

import pronouncing
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ──────────────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen3-32B"

CLEAN_PROMPT   = "A rhyming couplet:\nShe felt a sudden sense of fright,\nand hoped that"
CORRUPT_PROMPT = "A rhyming couplet:\nShe felt a sudden sense of fear,\nand hoped that"

CLEAN_RHYME_WORD   = "fright"
CORRUPT_RHYME_WORD = "fear"

SAMPLING_N    = 100
SAMPLING_TEMP = 0.8
MAX_NEW_TOKENS = 20

EXPERIMENTS = [
    {"experiment_id": "exp_fear_minus1", "selector": {"kind": "relative_to_newline", "offset": -1}},
    {"experiment_id": "exp_fear_minus2", "selector": {"kind": "relative_to_newline", "offset": -2}},
]

RESULTS_SUBDIR = "QWEN3_PER_LAYER"

# ── Rhyme Checking ───────────────────────────────────────────────────────────────

def _rhyme_score(w1: str, w2: str):
    p1 = pronouncing.phones_for_word(w1.lower().strip())
    p2 = pronouncing.phones_for_word(w2.lower().strip())
    if not p1 or not p2:
        return None
    rp1 = pronouncing.rhyming_part(p1[0])
    rp2 = pronouncing.rhyming_part(p2[0])
    return (rp1 == rp2) if (rp1 and rp2) else None

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

def rhyme_rate(completions: list, prompt: str, rhyme_word: str) -> float:
    hits = sum(
        1 for c in completions
        if _rhyme_score(extract_rhyme_word(c, prompt), rhyme_word) is True
    )
    return hits / len(completions) if completions else 0.0

# ── Model Loading ────────────────────────────────────────────────────────────────

def load_model():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Loaded. Layers: {model.config.num_hidden_layers} | d_model: {model.config.hidden_size}")
    return model, tokenizer

# ── Generation Helpers ───────────────────────────────────────────────────────────

def get_input_device(model):
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
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def sample_completions(model, tokenizer, prompt: str, n: int, temperature: float) -> list:
    return [generate_text(model, tokenizer, prompt, temperature) for _ in tqdm(range(n), desc="Sampling", leave=False)]

# ── Token Position Finding ───────────────────────────────────────────────────────

def find_patch_pos(tokenizer, prompt: str, selector: dict) -> tuple:
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    token_ids      = enc["input_ids"]
    offset_mapping = enc["offset_mapping"]

    if selector["kind"] == "newline":
        newline_char_positions = [i for i, ch in enumerate(prompt) if ch == "\n"]
        if not newline_char_positions:
            raise ValueError("No newline in prompt.")
        last_nl_char = newline_char_positions[-1]
        patch_pos = next(
            (i for i, (s, e) in enumerate(offset_mapping) if s <= last_nl_char < e),
            None,
        )
        if patch_pos is None:
            raise ValueError("Could not find token covering the last newline.")
        patch_label = f"newline (pos={patch_pos})"
        return patch_pos, patch_label

    if selector["kind"] == "token_text":
        target = selector["token_text"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        n = len(target_ids)
        last_match_end = None
        for i in range(len(token_ids) - n, -1, -1):
            if token_ids[i : i + n] == target_ids:
                last_match_end = i + n - 1
                break
        if last_match_end is not None:
            patch_pos = last_match_end
            patch_label = f"token '{target}' (pos={patch_pos})"
            return patch_pos, patch_label
        for i in range(len(token_ids) - 1, -1, -1):
            decoded = tokenizer.decode([token_ids[i]], skip_special_tokens=False)
            if decoded.strip() == target.strip():
                patch_pos = i
                patch_label = f"token '{target}' (pos={patch_pos})"
                return patch_pos, patch_label
        raise ValueError(f"Could not find token for '{target}' in prompt: {repr(prompt)}")

    if selector["kind"] == "relative_to_newline":
        offset = selector["offset"]
        newline_chars = [i for i, ch in enumerate(prompt) if ch == "\n"]
        if len(newline_chars) < 2:
            raise ValueError("Need at least 2 newlines in prompt.")
        second_nl_char = newline_chars[1]
        second_nl_tok = next(
            (i for i, (s, e) in enumerate(offset_mapping) if s <= second_nl_char < e),
            None,
        )
        if second_nl_tok is None:
            raise ValueError("Could not find token covering second newline.")
        patch_pos = second_nl_tok + offset
        if patch_pos < 0 or patch_pos >= len(token_ids):
            raise ValueError(f"offset={offset} gives out-of-bounds pos={patch_pos}")
        tok_str = tokenizer.decode([token_ids[patch_pos]])
        sign = "+" if offset >= 0 else ""
        patch_label = f"i={sign}{offset} (pos={patch_pos}, tok={repr(tok_str)})"
        return patch_pos, patch_label

    raise ValueError(f"Unknown selector kind: {selector['kind']}")

# ── Activation Caching ───────────────────────────────────────────────────────────

def cache_hidden_states_at_pos(model, tokenizer, prompt: str, patch_pos: int) -> list:
    layers = model.model.layers
    device = get_input_device(model)
    enc = tokenizer(prompt, return_tensors="pt").to(device)

    cached = [None] * len(layers)
    handles = []

    def make_capture(idx):
        def hook(module, args):
            h = args[0]
            if h.shape[1] > patch_pos:
                cached[idx] = h[:, patch_pos, :].detach().clone()
        return hook

    for idx, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(make_capture(idx)))

    try:
        with torch.no_grad():
            model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    missing = [i for i, v in enumerate(cached) if v is None]
    if missing:
        raise RuntimeError(f"Failed to cache hidden states for layers: {missing}")
    return cached

# ── Patching Context Manager ─────────────────────────────────────────────────────

@contextmanager
def patch_layer_at_pos(model, layer_idx: int, patch_pos: int, patch_vec: torch.Tensor):
    layer = model.model.layers[layer_idx]

    def hook(module, args):
        h = args[0]
        if h.shape[1] <= 1 or h.shape[1] <= patch_pos:
            return args
        out = h.clone()
        out[:, patch_pos, :] = patch_vec.to(out.device, dtype=out.dtype)
        return (out,) + args[1:]

    handle = layer.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()

# ── Single Experiment ────────────────────────────────────────────────────────────

def run_single_experiment(
    model,
    tokenizer,
    experiment_id: str,
    selector: dict,
    baseline_completions: list,
    baseline_clean_rate: float,
    baseline_corrupt_rate: float,
    clean_completion: str,
    corrupt_completion: str,
):
    n_layers = model.config.num_hidden_layers

    corrupt_patch_pos, corrupt_patch_label = find_patch_pos(tokenizer, CORRUPT_PROMPT, selector)
    clean_patch_pos,   clean_patch_label   = find_patch_pos(tokenizer, CLEAN_PROMPT,   selector)

    if corrupt_patch_pos != clean_patch_pos:
        print(f"  WARNING: patch position differs between prompts "
              f"(corrupt={corrupt_patch_pos}, clean={clean_patch_pos}). Using corrupt pos.")

    patch_pos = corrupt_patch_pos

    print(f"\n── Experiment: {experiment_id} | patch at {corrupt_patch_label} ──")

    enc = tokenizer(CORRUPT_PROMPT, return_tensors="pt")
    tok_list = enc["input_ids"][0].tolist()
    print("  Corrupt prompt tokens (±3 around patch pos):")
    for i in range(max(0, patch_pos - 3), min(len(tok_list), patch_pos + 4)):
        marker = " ← PATCH" if i == patch_pos else ""
        print(f"    pos {i:2d}: {repr(tokenizer.decode([tok_list[i]]))}{marker}")

    print(f"  Caching corrupt activations at pos={corrupt_patch_pos}...")
    corrupt_cache = cache_hidden_states_at_pos(model, tokenizer, CORRUPT_PROMPT, corrupt_patch_pos)

    print(f"  Sweeping {n_layers} layers (N={SAMPLING_N}, T={SAMPLING_TEMP})...")
    layer_results = []

    for layer_idx in tqdm(range(n_layers), desc=f"  {experiment_id}"):
        corrupt_vec = corrupt_cache[layer_idx]
        with patch_layer_at_pos(model, layer_idx, patch_pos, corrupt_vec):
            completions = sample_completions(model, tokenizer, CLEAN_PROMPT, SAMPLING_N, SAMPLING_TEMP)

        clean_rate   = rhyme_rate(completions, CLEAN_PROMPT, CLEAN_RHYME_WORD)
        corrupt_rate = rhyme_rate(completions, CLEAN_PROMPT, CORRUPT_RHYME_WORD)
        layer_results.append({
            "layer":             layer_idx,
            "completions":       completions,
            "clean_rhyme_rate":  clean_rate,
            "corrupt_rhyme_rate": corrupt_rate,
            "baseline_clean_rate":   baseline_clean_rate,
            "baseline_corrupt_rate": baseline_corrupt_rate,
        })

    best = max(layer_results, key=lambda r: r["corrupt_rhyme_rate"])
    print(f"  Best layer: {best['layer']} "
          f"(corrupt_rhyme_rate={best['corrupt_rhyme_rate']:.3f}, baseline={baseline_corrupt_rate:.3f})")

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", RESULTS_SUBDIR)
    run_name = f"qwen3_32b_{experiment_id}_corrupt_to_clean"
    run_dir  = os.path.join(results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    export = {
        "timestamp_utc":      datetime.now(timezone.utc).isoformat(),
        "experiment_id":      experiment_id,
        "run_name":           run_name,
        "model_name":         MODEL_NAME,
        "patch_source_mode":  "corrupt_to_clean",
        "target_selector":    selector,
        "target_patch_label": clean_patch_label,
        "target_patch_pos":   clean_patch_pos,
        "source_selector":    selector,
        "source_patch_label": corrupt_patch_label,
        "source_patch_pos":   corrupt_patch_pos,
        "sampling_mode":      True,
        "sampling_n":         SAMPLING_N,
        "sampling_temp":      SAMPLING_TEMP,
        "max_new_tokens":     MAX_NEW_TOKENS,
        "clean_prompt":       CLEAN_PROMPT,
        "corrupt_prompt":     CORRUPT_PROMPT,
        "clean_rhyme_word":   CLEAN_RHYME_WORD,
        "corrupt_rhyme_word": CORRUPT_RHYME_WORD,
        "n_layers":           n_layers,
        "baseline": {
            "clean_completion":   clean_completion,
            "corrupt_completion": corrupt_completion,
            "completions":        baseline_completions,
            "unpatched_clean_clean_rhyme_rate":   baseline_clean_rate,
            "unpatched_clean_corrupt_rhyme_rate": baseline_corrupt_rate,
        },
        "results": layer_results,
    }

    json_path = os.path.join(run_dir, "generations.json")
    with open(json_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"  Saved to {json_path}")

    return export

# ── Main ─────────────────────────────────────────────────────────────────────────

def run_all():
    model, tokenizer = load_model()

    clean_ids   = tokenizer(CLEAN_PROMPT,   return_tensors="pt").input_ids
    corrupt_ids = tokenizer(CORRUPT_PROMPT, return_tensors="pt").input_ids
    print(f"\nClean prompt token length:   {clean_ids.shape[1]}")
    print(f"Corrupt prompt token length: {corrupt_ids.shape[1]}")
    if clean_ids.shape[1] != corrupt_ids.shape[1]:
        print("WARNING: token length mismatch — positional alignment may be off.")

    print("\n── Greedy Baselines ──")
    clean_completion   = generate_text(model, tokenizer, CLEAN_PROMPT,   temperature=0)
    corrupt_completion = generate_text(model, tokenizer, CORRUPT_PROMPT, temperature=0)
    print(f"Clean   -> {repr(clean_completion)}")
    print(f"Corrupt -> {repr(corrupt_completion)}")

    print(f"\n── Unpatched Clean Baseline ({SAMPLING_N} samples, T={SAMPLING_TEMP}) ──")
    baseline_completions  = sample_completions(model, tokenizer, CLEAN_PROMPT, SAMPLING_N, SAMPLING_TEMP)
    baseline_clean_rate   = rhyme_rate(baseline_completions, CLEAN_PROMPT, CLEAN_RHYME_WORD)
    baseline_corrupt_rate = rhyme_rate(baseline_completions, CLEAN_PROMPT, CORRUPT_RHYME_WORD)
    print(f"  Rhymes with '{CLEAN_RHYME_WORD}'   (expected high): {baseline_clean_rate:.3f}")
    print(f"  Rhymes with '{CORRUPT_RHYME_WORD}' (expected low):  {baseline_corrupt_rate:.3f}")

    for exp in EXPERIMENTS:
        run_single_experiment(
            model=model,
            tokenizer=tokenizer,
            experiment_id=exp["experiment_id"],
            selector=exp["selector"],
            baseline_completions=baseline_completions,
            baseline_clean_rate=baseline_clean_rate,
            baseline_corrupt_rate=baseline_corrupt_rate,
            clean_completion=clean_completion,
            corrupt_completion=corrupt_completion,
        )

    print("\n── All experiments complete. ──")


if __name__ == "__main__":
    run_all()
