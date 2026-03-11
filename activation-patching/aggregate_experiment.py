"""
Aggregate activation-patching experiment across 4 prompt pairs.

Patch direction: corrupt -> clean
  - Cache CORRUPT activations at the last newline position
  - Run CLEAN prompt with CORRUPT activation patched in at each layer
  - Measure corrupt_rhyme_rate: does the clean run now rhyme with the corrupt word?

Per-pair and aggregate (mean across pairs) results are saved as JSON.
"""

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone

import pronouncing
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# -- Config ----------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-32B"

SAMPLING_N    = 20
SAMPLING_TEMP = 0.7
MAX_NEW_TOKENS = 20

PROMPT_PAIRS = [
    {
        "pair_id":            "dread_gloom",
        "clean_prompt":       "A rhyming couplet:\nThe empty house was filled with silent dread,\nuntil the moment",
        "corrupt_prompt":     "A rhyming couplet:\nThe empty house was filled with silent gloom,\nuntil the moment",
        "clean_rhyme_word":   "dread",
        "corrupt_rhyme_word": "gloom",
    },
    {
        "pair_id":            "bliss_joy",
        "clean_prompt":       "A rhyming couplet:\nThe children laughed in bliss,\nuntil they all",
        "corrupt_prompt":     "A rhyming couplet:\nThe children laughed in joy,\nuntil they all",
        "clean_rhyme_word":   "bliss",
        "corrupt_rhyme_word": "joy",
    },
    {
        "pair_id":            "night_dark",
        "clean_prompt":       "A rhyming couplet:\nShe wandered home alone into the night,\nand then she",
        "corrupt_prompt":     "A rhyming couplet:\nShe wandered home alone into the dark,\nand then she",
        "clean_rhyme_word":   "night",
        "corrupt_rhyme_word": "dark",
    },
    {
        "pair_id":            "grief_pain",
        "clean_prompt":       "A rhyming couplet:\nI never knew the depth of such a grief,\nuntil the world",
        "corrupt_prompt":     "A rhyming couplet:\nI never knew the depth of such a pain,\nuntil the world",
        "clean_rhyme_word":   "grief",
        "corrupt_rhyme_word": "pain",
    },
]

# -- Rhyme Helpers ---------------------------------------------------------------

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

# -- Model Loading ---------------------------------------------------------------

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

# -- Generation Helpers ----------------------------------------------------------

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
    return [
        generate_text(model, tokenizer, prompt, temperature)
        for _ in tqdm(range(n), desc="Sampling", leave=False)
    ]

# -- Token Position Finding ------------------------------------------------------

def find_newline_patch_pos(tokenizer, prompt: str) -> tuple:
    """Find the last token covering the last newline character in the prompt."""
    last_nl_char = max(i for i, ch in enumerate(prompt) if ch == "\n")
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    offset_mapping = enc["offset_mapping"]
    patch_pos = next(
        (i for i, (s, e) in enumerate(offset_mapping) if s <= last_nl_char < e),
        None,
    )
    if patch_pos is None:
        raise ValueError(f"Could not find token covering last newline in: {repr(prompt)}")
    return patch_pos, f"newline (pos={patch_pos})"

# -- Activation Caching ----------------------------------------------------------

def cache_hidden_states_at_pos(model, tokenizer, prompt: str, patch_pos: int) -> list:
    """Cache resid_pre at patch_pos for every layer via forward_pre_hooks."""
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

# -- Patching Context Manager ----------------------------------------------------

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

# -- Single Pair Experiment ------------------------------------------------------

def run_pair(model, tokenizer, pair: dict, results_dir: str) -> dict:
    pair_id            = pair["pair_id"]
    clean_prompt       = pair["clean_prompt"]
    corrupt_prompt     = pair["corrupt_prompt"]
    clean_rhyme_word   = pair["clean_rhyme_word"]
    corrupt_rhyme_word = pair["corrupt_rhyme_word"]
    n_layers           = model.config.num_hidden_layers

    print(f"\n{'='*60}")
    print(f"Pair: {pair_id}  ({clean_rhyme_word!r} vs {corrupt_rhyme_word!r})")
    print(f"{'='*60}")

    # Check token lengths
    clean_ids   = tokenizer(clean_prompt,   return_tensors="pt").input_ids
    corrupt_ids = tokenizer(corrupt_prompt, return_tensors="pt").input_ids
    if clean_ids.shape[1] != corrupt_ids.shape[1]:
        print(f"  WARNING: token length mismatch ({clean_ids.shape[1]} vs {corrupt_ids.shape[1]})")

    # Patch position: last newline in each prompt
    corrupt_patch_pos, corrupt_patch_label = find_newline_patch_pos(tokenizer, corrupt_prompt)
    clean_patch_pos,   clean_patch_label   = find_newline_patch_pos(tokenizer, clean_prompt)
    if corrupt_patch_pos != clean_patch_pos:
        print(f"  WARNING: newline pos differs (corrupt={corrupt_patch_pos}, clean={clean_patch_pos}). Using corrupt pos.")
    patch_pos = corrupt_patch_pos

    # Show tokens around patch position
    tok_list = corrupt_ids[0].tolist()
    print(f"  Corrupt prompt tokens (+-3 around patch pos={patch_pos}):")
    for i in range(max(0, patch_pos - 3), min(len(tok_list), patch_pos + 4)):
        marker = " <- PATCH" if i == patch_pos else ""
        print(f"    pos {i:2d}: {repr(tokenizer.decode([tok_list[i]]))}{marker}")

    # Greedy baselines
    print("  Greedy baselines...")
    clean_completion   = generate_text(model, tokenizer, clean_prompt,   temperature=0)
    corrupt_completion = generate_text(model, tokenizer, corrupt_prompt, temperature=0)
    print(f"  Clean   -> {repr(clean_completion[:80])}")
    print(f"  Corrupt -> {repr(corrupt_completion[:80])}")

    # Sampling baseline: unpatched CLEAN run
    print(f"  Unpatched clean baseline (N={SAMPLING_N}, T={SAMPLING_TEMP})...")
    baseline_completions  = sample_completions(model, tokenizer, clean_prompt, SAMPLING_N, SAMPLING_TEMP)
    baseline_clean_rate   = rhyme_rate(baseline_completions, clean_prompt, clean_rhyme_word)
    baseline_corrupt_rate = rhyme_rate(baseline_completions, clean_prompt, corrupt_rhyme_word)
    print(f"    Rhymes with '{clean_rhyme_word}' (expected high): {baseline_clean_rate:.3f}")
    print(f"    Rhymes with '{corrupt_rhyme_word}' (expected low):  {baseline_corrupt_rate:.3f}")

    # Cache CORRUPT activations at patch_pos
    print(f"  Caching corrupt activations at pos={corrupt_patch_pos}...")
    corrupt_cache = cache_hidden_states_at_pos(model, tokenizer, corrupt_prompt, corrupt_patch_pos)

    # Layer sweep: run CLEAN with CORRUPT activation patched in
    print(f"  Sweeping {n_layers} layers...")
    layer_results = []

    for layer_idx in tqdm(range(n_layers), desc=f"  {pair_id}"):
        corrupt_vec = corrupt_cache[layer_idx]
        with patch_layer_at_pos(model, layer_idx, patch_pos, corrupt_vec):
            completions = sample_completions(model, tokenizer, clean_prompt, SAMPLING_N, SAMPLING_TEMP)

        clean_rate   = rhyme_rate(completions, clean_prompt, clean_rhyme_word)
        corrupt_rate = rhyme_rate(completions, clean_prompt, corrupt_rhyme_word)
        layer_results.append({
            "layer":                 layer_idx,
            "completions":           completions,
            "clean_rhyme_rate":      clean_rate,
            "corrupt_rhyme_rate":    corrupt_rate,
            "baseline_clean_rate":   baseline_clean_rate,
            "baseline_corrupt_rate": baseline_corrupt_rate,
            "delta_corrupt_rate":    corrupt_rate - baseline_corrupt_rate,
        })

    best = max(layer_results, key=lambda r: r["corrupt_rhyme_rate"])
    print(f"  Best layer: {best['layer']} (corrupt_rhyme_rate={best['corrupt_rhyme_rate']:.3f}, baseline={baseline_corrupt_rate:.3f})")

    # Save per-pair JSON
    pair_dir = os.path.join(results_dir, pair_id)
    os.makedirs(pair_dir, exist_ok=True)
    export = {
        "timestamp_utc":       datetime.now(timezone.utc).isoformat(),
        "pair_id":             pair_id,
        "model_name":          MODEL_NAME,
        "patch_direction":     "corrupt->clean",
        "patch_pos":           patch_pos,
        "patch_label":         corrupt_patch_label,
        "sampling_n":          SAMPLING_N,
        "sampling_temp":       SAMPLING_TEMP,
        "max_new_tokens":      MAX_NEW_TOKENS,
        "clean_prompt":        clean_prompt,
        "corrupt_prompt":      corrupt_prompt,
        "clean_rhyme_word":    clean_rhyme_word,
        "corrupt_rhyme_word":  corrupt_rhyme_word,
        "n_layers":            n_layers,
        "baseline": {
            "clean_completion":   clean_completion,
            "corrupt_completion": corrupt_completion,
            "completions":        baseline_completions,
            "unpatched_clean_clean_rhyme_rate":   baseline_clean_rate,
            "unpatched_clean_corrupt_rhyme_rate": baseline_corrupt_rate,
        },
        "results": layer_results,
    }
    json_path = os.path.join(pair_dir, "generations.json")
    with open(json_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"  Saved to {json_path}")

    return export

# -- Aggregate -------------------------------------------------------------------

def compute_aggregate(all_pair_exports: list) -> list:
    """Average corrupt_rhyme_rate and delta_corrupt_rate per layer across all pairs."""
    n_layers = all_pair_exports[0]["n_layers"]
    aggregate = []
    for layer_idx in range(n_layers):
        corrupt_rates = [
            exp["results"][layer_idx]["corrupt_rhyme_rate"]
            for exp in all_pair_exports
        ]
        delta_rates = [
            exp["results"][layer_idx]["delta_corrupt_rate"]
            for exp in all_pair_exports
        ]
        aggregate.append({
            "layer":                      layer_idx,
            "mean_corrupt_rhyme_rate":    sum(corrupt_rates) / len(corrupt_rates),
            "mean_delta_corrupt_rate":    sum(delta_rates) / len(delta_rates),
            "per_pair_corrupt_rhyme_rate": {
                exp["pair_id"]: exp["results"][layer_idx]["corrupt_rhyme_rate"]
                for exp in all_pair_exports
            },
        })
    return aggregate

# -- Main ------------------------------------------------------------------------

def run_all():
    model, tokenizer = load_model()

    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        "QWEN3_AGGREGATE",
        f"qwen3_32b_aggregate_N{SAMPLING_N}",
    )
    os.makedirs(results_dir, exist_ok=True)

    all_pair_exports = []
    for pair in PROMPT_PAIRS:
        export = run_pair(model, tokenizer, pair, results_dir)
        all_pair_exports.append(export)

    # Compute and save aggregate
    aggregate = compute_aggregate(all_pair_exports)
    best = max(aggregate, key=lambda r: r["mean_corrupt_rhyme_rate"])
    print(f"\n-- Aggregate Summary --")
    print(f"Best layer: {best['layer']} (mean_corrupt_rhyme_rate={best['mean_corrupt_rhyme_rate']:.3f})")
    for r in aggregate:
        print(f"  Layer {r['layer']:2d}: mean_corrupt_rate={r['mean_corrupt_rhyme_rate']:.3f}  mean_delta={r['mean_delta_corrupt_rate']:+.3f}")

    agg_path = os.path.join(results_dir, "aggregate.json")
    with open(agg_path, "w") as f:
        json.dump({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_name":    MODEL_NAME,
            "sampling_n":    SAMPLING_N,
            "sampling_temp": SAMPLING_TEMP,
            "pairs":         [p["pair_id"] for p in PROMPT_PAIRS],
            "aggregate":     aggregate,
        }, f, indent=2)
    print(f"\nAggregate saved to {agg_path}")


if __name__ == "__main__":
    run_all()
