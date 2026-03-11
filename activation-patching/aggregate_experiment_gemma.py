"""
Aggregate activation-patching experiment across 4 prompt pairs — Gemma-3-27B-IT.

Patch direction: corrupt -> clean
  - Cache CORRUPT activations at each patch position
  - Run CLEAN prompt with CORRUPT activation patched in at each layer
  - Measure corrupt_rhyme_rate: does the clean run now rhyme with the corrupt word?

Sweeps 6 positions relative to the second newline (i=-2,-1,0,+1,+2,+3)
and all layers for each position × pair combination.
"""

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone

import pronouncing
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# -- Config ----------------------------------------------------------------------

MODEL_NAME    = "google/gemma-3-27b-it"
SAMPLING_N    = 20
SAMPLING_TEMP = 0.7
MAX_NEW_TOKENS = 20

POSITIONS = [
    {"offset": -2, "pos_id": "i_minus2"},
    {"offset": -1, "pos_id": "i_minus1"},
    {"offset":  0, "pos_id": "i_0"},
    {"offset": +1, "pos_id": "i_plus1"},
    {"offset": +2, "pos_id": "i_plus2"},
    {"offset": +3, "pos_id": "i_plus3"},
]

PROMPT_PAIRS = [
    {
        "pair_id":            "doom_dread",
        "clean_prompt":       "A rhyming couplet:\nThe empty house was filled with silent doom,\nwhen suddenly they",
        "corrupt_prompt":     "A rhyming couplet:\nThe empty house was filled with silent dread,\nwhen suddenly they",
        "clean_rhyme_word":   "doom",
        "corrupt_rhyme_word": "dread",
    },
    {
        "pair_id":            "bliss_joy",
        "clean_prompt":       "A rhyming couplet:\nThe children laughed in bliss,\nuntil they all",
        "corrupt_prompt":     "A rhyming couplet:\nThe children laughed in joy,\nuntil they all",
        "clean_rhyme_word":   "bliss",
        "corrupt_rhyme_word": "joy",
    },
    {
        "pair_id":            "dark_night",
        "clean_prompt":       "A rhyming couplet:\nShe wandered home alone into the dark,\nand then she",
        "corrupt_prompt":     "A rhyming couplet:\nShe wandered home alone into the night,\nand then she",
        "clean_rhyme_word":   "dark",
        "corrupt_rhyme_word": "night",
    },
    {
        "pair_id":            "grief_pain",
        "clean_prompt":       "A rhyming couplet:\nI never knew the depth of such grief,\nas though the",
        "corrupt_prompt":     "A rhyming couplet:\nI never knew the depth of such pain,\nas though the",
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

# -- Model (Gemma-specific) ------------------------------------------------------

def load_model():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    from transformers import Gemma3ForConditionalGeneration
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    text_cfg = model.config.text_config
    print(f"Loaded. Layers: {text_cfg.num_hidden_layers} | d_model: {text_cfg.hidden_size}")
    return model, tokenizer

def get_input_device(model):
    return model.model.language_model.embed_tokens.weight.device

def get_layers(model):
    return model.model.language_model.layers

def get_n_layers(model):
    return model.config.text_config.num_hidden_layers

# -- Generation Helpers ----------------------------------------------------------

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

def find_patch_pos(tokenizer, prompt: str, offset: int) -> tuple:
    """
    Find token position at `offset` relative to the second newline (i=0).
    Returns (patch_pos, patch_label, tok_str).
    """
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    token_ids      = enc["input_ids"]
    offset_mapping = enc["offset_mapping"]

    newline_chars = [i for i, ch in enumerate(prompt) if ch == "\n"]
    if len(newline_chars) < 2:
        raise ValueError(f"Need at least 2 newlines in prompt, found {len(newline_chars)}.")
    second_nl_char = newline_chars[1]

    second_nl_tok = next(
        (i for i, (s, e) in enumerate(offset_mapping) if s <= second_nl_char < e),
        None,
    )
    if second_nl_tok is None:
        raise ValueError("Could not find token covering the second newline.")

    patch_pos = second_nl_tok + offset
    if patch_pos < 0 or patch_pos >= len(token_ids):
        raise ValueError(f"offset={offset} gives out-of-bounds pos={patch_pos} (prompt has {len(token_ids)} tokens)")

    tok_str = tokenizer.decode([token_ids[patch_pos]])
    sign = "+" if offset >= 0 else ""
    patch_label = f"i={sign}{offset} (pos={patch_pos}, tok={repr(tok_str)})"
    return patch_pos, patch_label, tok_str

# -- Activation Caching ----------------------------------------------------------

def cache_hidden_states_at_pos(model, tokenizer, prompt: str, patch_pos: int) -> list:
    layers = get_layers(model)
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
    layer = get_layers(model)[layer_idx]

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

# -- Single Position × Pair Run --------------------------------------------------

def run_position(model, tokenizer, pair: dict, offset: int, pos_id: str,
                 baseline_completions: list, baseline_clean_rate: float,
                 baseline_corrupt_rate: float, pair_dir: str) -> dict:
    clean_prompt       = pair["clean_prompt"]
    corrupt_prompt     = pair["corrupt_prompt"]
    clean_rhyme_word   = pair["clean_rhyme_word"]
    corrupt_rhyme_word = pair["corrupt_rhyme_word"]
    n_layers           = get_n_layers(model)

    corrupt_patch_pos, corrupt_patch_label, _ = find_patch_pos(tokenizer, corrupt_prompt, offset)
    clean_patch_pos,   clean_patch_label,   _ = find_patch_pos(tokenizer, clean_prompt,   offset)

    if corrupt_patch_pos != clean_patch_pos:
        print(f"    WARNING: patch pos differs (corrupt={corrupt_patch_pos}, clean={clean_patch_pos}). Using corrupt pos.")
    patch_pos = corrupt_patch_pos

    print(f"  Position {pos_id} | patch at {corrupt_patch_label}")

    tok_list = tokenizer(corrupt_prompt, add_special_tokens=True).input_ids
    ctx = "  ".join(
        repr(tokenizer.decode([tok_list[i]]) + (" <-" if i == patch_pos else ""))
        for i in range(max(0, patch_pos - 2), min(len(tok_list), patch_pos + 3))
    )
    print(f"    context: {ctx}")

    corrupt_cache = cache_hidden_states_at_pos(model, tokenizer, corrupt_prompt, corrupt_patch_pos)

    layer_results = []
    for layer_idx in tqdm(range(n_layers), desc=f"    layers", leave=False):
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
    print(f"    Best layer: {best['layer']} (corrupt_rhyme_rate={best['corrupt_rhyme_rate']:.3f}, baseline={baseline_corrupt_rate:.3f})")

    export = {
        "timestamp_utc":       datetime.now(timezone.utc).isoformat(),
        "pair_id":             pair["pair_id"],
        "pos_id":              pos_id,
        "offset":              offset,
        "model_name":          MODEL_NAME,
        "patch_direction":     "corrupt->clean",
        "corrupt_patch_pos":   corrupt_patch_pos,
        "corrupt_patch_label": corrupt_patch_label,
        "clean_patch_pos":     clean_patch_pos,
        "clean_patch_label":   clean_patch_label,
        "sampling_n":          SAMPLING_N,
        "sampling_temp":       SAMPLING_TEMP,
        "max_new_tokens":      MAX_NEW_TOKENS,
        "clean_prompt":        clean_prompt,
        "corrupt_prompt":      corrupt_prompt,
        "clean_rhyme_word":    clean_rhyme_word,
        "corrupt_rhyme_word":  corrupt_rhyme_word,
        "n_layers":            n_layers,
        "baseline": {
            "completions":        baseline_completions,
            "clean_rhyme_rate":   baseline_clean_rate,
            "corrupt_rhyme_rate": baseline_corrupt_rate,
        },
        "results": layer_results,
    }

    pos_dir = os.path.join(pair_dir, pos_id)
    os.makedirs(pos_dir, exist_ok=True)
    json_path = os.path.join(pos_dir, "generations.json")
    with open(json_path, "w") as f:
        json.dump(export, f, indent=2)

    return export

# -- Single Pair -----------------------------------------------------------------

def run_pair(model, tokenizer, pair: dict, results_dir: str) -> dict:
    pair_id            = pair["pair_id"]
    clean_prompt       = pair["clean_prompt"]
    corrupt_prompt     = pair["corrupt_prompt"]
    clean_rhyme_word   = pair["clean_rhyme_word"]
    corrupt_rhyme_word = pair["corrupt_rhyme_word"]

    print(f"\n{'='*60}")
    print(f"Pair: {pair_id}  ({clean_rhyme_word!r} vs {corrupt_rhyme_word!r})")
    print(f"{'='*60}")

    clean_ids   = tokenizer(clean_prompt,   return_tensors="pt").input_ids
    corrupt_ids = tokenizer(corrupt_prompt, return_tensors="pt").input_ids
    if clean_ids.shape[1] != corrupt_ids.shape[1]:
        print(f"  WARNING: token length mismatch ({clean_ids.shape[1]} vs {corrupt_ids.shape[1]})")

    print("  Greedy baselines...")
    clean_completion   = generate_text(model, tokenizer, clean_prompt,   temperature=0)
    corrupt_completion = generate_text(model, tokenizer, corrupt_prompt, temperature=0)
    print(f"  Clean   -> {repr(clean_completion[:100])}")
    print(f"  Corrupt -> {repr(corrupt_completion[:100])}")

    print(f"  Unpatched clean baseline (N={SAMPLING_N}, T={SAMPLING_TEMP})...")
    baseline_completions  = sample_completions(model, tokenizer, clean_prompt, SAMPLING_N, SAMPLING_TEMP)
    baseline_clean_rate   = rhyme_rate(baseline_completions, clean_prompt, clean_rhyme_word)
    baseline_corrupt_rate = rhyme_rate(baseline_completions, clean_prompt, corrupt_rhyme_word)
    print(f"    Rhymes with '{clean_rhyme_word}' (expected high): {baseline_clean_rate:.3f}")
    print(f"    Rhymes with '{corrupt_rhyme_word}' (expected low): {baseline_corrupt_rate:.3f}")

    pair_dir = os.path.join(results_dir, pair_id)
    os.makedirs(pair_dir, exist_ok=True)

    pair_meta = {
        "pair_id":               pair_id,
        "clean_prompt":          clean_prompt,
        "corrupt_prompt":        corrupt_prompt,
        "clean_rhyme_word":      clean_rhyme_word,
        "corrupt_rhyme_word":    corrupt_rhyme_word,
        "clean_completion":      clean_completion,
        "corrupt_completion":    corrupt_completion,
        "baseline_clean_rate":   baseline_clean_rate,
        "baseline_corrupt_rate": baseline_corrupt_rate,
    }
    with open(os.path.join(pair_dir, "pair_meta.json"), "w") as f:
        json.dump(pair_meta, f, indent=2)

    position_exports = {}
    for pos in POSITIONS:
        export = run_position(
            model, tokenizer, pair,
            offset=pos["offset"], pos_id=pos["pos_id"],
            baseline_completions=baseline_completions,
            baseline_clean_rate=baseline_clean_rate,
            baseline_corrupt_rate=baseline_corrupt_rate,
            pair_dir=pair_dir,
        )
        position_exports[pos["pos_id"]] = export

    return position_exports

# -- Aggregate -------------------------------------------------------------------

def compute_aggregate(all_position_exports: list) -> dict:
    pos_ids  = [p["pos_id"] for p in POSITIONS]
    n_layers = list(all_position_exports[0].values())[0]["n_layers"]

    aggregate = {}
    for pos_id in pos_ids:
        layer_agg = []
        for layer_idx in range(n_layers):
            corrupt_rates = [
                exports[pos_id]["results"][layer_idx]["corrupt_rhyme_rate"]
                for exports in all_position_exports
            ]
            delta_rates = [
                exports[pos_id]["results"][layer_idx]["delta_corrupt_rate"]
                for exports in all_position_exports
            ]
            layer_agg.append({
                "layer":                      layer_idx,
                "mean_corrupt_rhyme_rate":    sum(corrupt_rates) / len(corrupt_rates),
                "mean_delta_corrupt_rate":    sum(delta_rates)   / len(delta_rates),
                "per_pair_corrupt_rhyme_rate": {
                    exports[pos_id]["pair_id"]: exports[pos_id]["results"][layer_idx]["corrupt_rhyme_rate"]
                    for exports in all_position_exports
                },
            })
        aggregate[pos_id] = layer_agg

    return aggregate

# -- Main ------------------------------------------------------------------------

def run_all():
    model, tokenizer = load_model()

    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        "GEMMA3_AGGREGATE",
        f"gemma3_27b_aggregate_N{SAMPLING_N}",
    )
    os.makedirs(results_dir, exist_ok=True)

    all_position_exports = []
    for pair in PROMPT_PAIRS:
        position_exports = run_pair(model, tokenizer, pair, results_dir)
        all_position_exports.append(position_exports)

    aggregate = compute_aggregate(all_position_exports)

    print("\n-- Aggregate Summary --")
    for pos in POSITIONS:
        pos_id = pos["pos_id"]
        best = max(aggregate[pos_id], key=lambda r: r["mean_corrupt_rhyme_rate"])
        print(f"  {pos_id}: best layer={best['layer']} mean_corrupt_rate={best['mean_corrupt_rhyme_rate']:.3f}")

    agg_path = os.path.join(results_dir, "aggregate.json")
    with open(agg_path, "w") as f:
        json.dump({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_name":    MODEL_NAME,
            "sampling_n":    SAMPLING_N,
            "sampling_temp": SAMPLING_TEMP,
            "pairs":         [p["pair_id"] for p in PROMPT_PAIRS],
            "positions":     [p["pos_id"]  for p in POSITIONS],
            "aggregate":     aggregate,
        }, f, indent=2)
    print(f"\nAggregate saved to {agg_path}")


if __name__ == "__main__":
    run_all()
