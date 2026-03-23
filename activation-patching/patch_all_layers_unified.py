"""
Unified ALL-LAYERS activation-patching experiment across all model families.

Patches ALL layers simultaneously at a single token position (corrupt → clean).
Sweeps 6 positions relative to the second newline (i=-2,-1,0,+1,+2,+3).
Runs 4 prompt pairs, 100 samples each.

Produces 6 aggregate data points per model (one per position), averaged over
the 4 prompt pairs — compared to aggregate_experiment_unified.py which
produces n_layers × 6 points.

Supports:
  - Qwen3 (0.6B–32B):           standard AutoModelForCausalLM, model.model.layers
  - Gemma-3-1B-IT:               Gemma3ForCausalLM (text-only), model.model.layers
  - Gemma-3-4B/12B/27B-IT:       Gemma3ForConditionalGeneration (multimodal wrapper),
                                  model.model.language_model.layers
  - Llama-3.x (Instruct):        standard AutoModelForCausalLM, model.model.layers

Usage:
    python patch_all_layers_unified.py --model Qwen/Qwen3-8B
    python patch_all_layers_unified.py --model google/gemma-3-4b-it
    python patch_all_layers_unified.py --model meta-llama/Llama-3.1-70B-Instruct
    python patch_all_layers_unified.py --list-models
"""

import argparse
import json
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone

import pronouncing
import torch
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

SAMPLING_N     = 20
SAMPLING_TEMP  = 0.7
MAX_NEW_TOKENS = 20
BATCH_SIZE     = 100

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
    {
        "pair_id":            "fright_fear",
        "clean_prompt":       "A rhyming couplet:\nShe felt a sudden sense of fright,\nand hoped that",
        "corrupt_prompt":     "A rhyming couplet:\nShe felt a sudden sense of fear,\nand hoped that",
        "clean_rhyme_word":   "fright",
        "corrupt_rhyme_word": "fear",
    },
]

SUPPORTED_MODELS = {
    # Qwen3 family (base models)
    "Qwen/Qwen3-0.6B":   "qwen3_0.6b",
    "Qwen/Qwen3-1.7B":   "qwen3_1.7b",
    "Qwen/Qwen3-4B":     "qwen3_4b",
    "Qwen/Qwen3-8B":     "qwen3_8b",
    "Qwen/Qwen3-14B":    "qwen3_14b",
    # Gemma-3 family (instruct)
    "google/gemma-3-1b-it":  "gemma3_1b",
    "google/gemma-3-4b-it":  "gemma3_4b",
    "google/gemma-3-12b-it": "gemma3_12b",
    # Llama family (instruct)
    "meta-llama/Llama-3.2-1B-Instruct":  "llama3.2_1b_instruct",
    "meta-llama/Llama-3.2-3B-Instruct":  "llama3.2_3b_instruct",
    "meta-llama/Llama-3.1-8B-Instruct":  "llama3.1_8b_instruct",
    "meta-llama/Llama-3.1-70B-Instruct": "llama3.1_70b_instruct",
}

# ── Rhyme Helpers ──────────────────────────────────────────────────────────────

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

# ── Architecture Adapter ───────────────────────────────────────────────────────

class ModelAdapter:
    """
    Abstracts over architectural differences:
      - Gemma3ForConditionalGeneration (4B/12B/27B): layers under model.language_model
      - All others (Qwen, Llama, Gemma-3-1B):        layers under model
    """

    def __init__(self, model):
        self.model = model
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            # Gemma3ForConditionalGeneration (4B, 12B, 27B)
            self._layers_fn    = lambda m: m.model.language_model.layers
            self._device_fn    = lambda m: m.model.language_model.embed_tokens.weight.device
            self._n_layers_fn  = lambda m: m.config.text_config.num_hidden_layers
        elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            # Standard decoder-only: Qwen3, Llama, Gemma-3-1B (Gemma3ForCausalLM)
            self._layers_fn    = lambda m: m.model.layers
            self._device_fn    = lambda m: m.model.embed_tokens.weight.device
            self._n_layers_fn  = lambda m: m.config.num_hidden_layers
        else:
            raise RuntimeError(
                f"Unrecognized model architecture for {type(model).__name__}. "
                "Expected model.model.language_model (Gemma multimodal) or "
                "model.model.embed_tokens (standard decoder-only)."
            )

    def get_layers(self):
        return self._layers_fn(self.model)

    def get_input_device(self):
        return self._device_fn(self.model)

    def get_n_layers(self) -> int:
        return self._n_layers_fn(self.model)

# ── Model Loading ──────────────────────────────────────────────────────────────

def load_model(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_type_uses_conditional_gen = False
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if getattr(cfg, "model_type", "") == "gemma3":
            model_type_uses_conditional_gen = True
    except Exception:
        pass

    if model_type_uses_conditional_gen:
        from transformers import Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    adapter = ModelAdapter(model)
    print(f"Loaded. Class={type(model).__name__} | Layers={adapter.get_n_layers()} | Device={adapter.get_input_device()}")
    return model, tokenizer, adapter

# ── Generation ─────────────────────────────────────────────────────────────────

def generate_text(model, tokenizer, adapter: ModelAdapter, prompt: str, temperature: float) -> str:
    device = adapter.get_input_device()
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

def sample_completions(model, tokenizer, adapter: ModelAdapter,
                       prompt: str, n: int, temperature: float,
                       batch_size: int = BATCH_SIZE) -> list:
    device = adapter.get_input_device()
    enc = tokenizer(prompt, return_tensors="pt").to(device)

    completions = []
    remaining = n
    with tqdm(total=n, desc="Sampling", leave=False) as pbar:
        while remaining > 0:
            this_batch = min(remaining, batch_size)
            with torch.no_grad():
                output_ids = model.generate(
                    **enc,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=tokenizer.pad_token_id,
                    num_return_sequences=this_batch,
                )
            for row in output_ids:
                completions.append(tokenizer.decode(row, skip_special_tokens=True))
            remaining -= this_batch
            pbar.update(this_batch)
    return completions

# ── Token Position Finding ─────────────────────────────────────────────────────

def find_patch_pos(tokenizer, prompt: str, offset: int) -> tuple:
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
        raise ValueError(
            f"offset={offset} gives out-of-bounds pos={patch_pos} "
            f"(prompt has {len(token_ids)} tokens)"
        )

    tok_str = tokenizer.decode([token_ids[patch_pos]])
    sign = "+" if offset >= 0 else ""
    patch_label = f"i={sign}{offset} (pos={patch_pos}, tok={repr(tok_str)})"
    return patch_pos, patch_label, tok_str

# ── Activation Caching ─────────────────────────────────────────────────────────

def cache_hidden_states_at_pos(model, tokenizer, adapter: ModelAdapter,
                                prompt: str, patch_pos: int) -> list:
    """Cache resid_pre at patch_pos for every layer via forward_pre_hooks."""
    layers = adapter.get_layers()
    device = adapter.get_input_device()
    enc = tokenizer(prompt, return_tensors="pt").to(device)

    cached  = [None] * len(layers)
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

# ── All-Layers Patch Hook ──────────────────────────────────────────────────────

def make_patch_hook(patch_vec: torch.Tensor, patch_pos: int):
    def hook_fn(module, args):
        h = args[0]
        if h.shape[1] <= 1 or h.shape[1] <= patch_pos:
            return args
        out = h.clone()
        out[:, patch_pos, :] = patch_vec.to(out.device, dtype=out.dtype)
        return (out,) + args[1:]
    return hook_fn

# ── Position Sweep (all layers patched simultaneously) ────────────────────────

def run_position(model, tokenizer, adapter: ModelAdapter,
                 model_name: str,
                 pair: dict, offset: int, pos_id: str,
                 baseline_completions: list,
                 baseline_clean_rate: float,
                 baseline_corrupt_rate: float,
                 pair_dir: str,
                 batch_size: int = BATCH_SIZE) -> dict:

    clean_prompt       = pair["clean_prompt"]
    corrupt_prompt     = pair["corrupt_prompt"]
    clean_rhyme_word   = pair["clean_rhyme_word"]
    corrupt_rhyme_word = pair["corrupt_rhyme_word"]
    n_layers           = adapter.get_n_layers()

    # resume: skip if already done
    pos_dir   = os.path.join(pair_dir, pos_id)
    json_path = os.path.join(pos_dir, "generations.json")
    if os.path.exists(json_path):
        print(f"  Position {pos_id} — already done, loading from disk.")
        with open(json_path) as f:
            return json.load(f)

    corrupt_patch_pos, corrupt_patch_label, _ = find_patch_pos(tokenizer, corrupt_prompt, offset)
    clean_patch_pos,   clean_patch_label,   _ = find_patch_pos(tokenizer, clean_prompt,   offset)

    if corrupt_patch_pos != clean_patch_pos:
        print(f"  WARNING: patch pos differs (corrupt={corrupt_patch_pos}, "
              f"clean={clean_patch_pos}). Using corrupt pos.")
    patch_pos = corrupt_patch_pos

    print(f"  Position {pos_id} | patch at {corrupt_patch_label} | all {n_layers} layers")

    tok_list = tokenizer(corrupt_prompt, add_special_tokens=True).input_ids
    ctx = "  ".join(
        repr(tokenizer.decode([tok_list[i]]) + (" <-" if i == patch_pos else ""))
        for i in range(max(0, patch_pos - 2), min(len(tok_list), patch_pos + 3))
    )
    print(f"    context: {ctx}")

    # Cache corrupt activations (resid_pre) at patch_pos for every layer
    corrupt_cache = cache_hidden_states_at_pos(
        model, tokenizer, adapter, corrupt_prompt, corrupt_patch_pos
    )

    # Register hooks on ALL layers simultaneously
    layers = adapter.get_layers()
    handles = []
    for layer_idx in range(n_layers):
        patch_vec = corrupt_cache[layer_idx]
        handle = layers[layer_idx].register_forward_pre_hook(
            make_patch_hook(patch_vec, patch_pos)
        )
        handles.append(handle)

    try:
        completions = sample_completions(
            model, tokenizer, adapter, clean_prompt, SAMPLING_N, SAMPLING_TEMP,
            batch_size=batch_size,
        )
    finally:
        for h in handles:
            h.remove()

    clean_rate   = rhyme_rate(completions, clean_prompt, clean_rhyme_word)
    corrupt_rate = rhyme_rate(completions, clean_prompt, corrupt_rhyme_word)
    delta        = corrupt_rate - baseline_corrupt_rate

    print(f"    corrupt_rhyme_rate={corrupt_rate:.3f}  "
          f"clean_rhyme_rate={clean_rate:.3f}  "
          f"delta={delta:+.3f}  (baseline_corrupt={baseline_corrupt_rate:.3f})")

    export = {
        "timestamp_utc":         datetime.now(timezone.utc).isoformat(),
        "pair_id":               pair["pair_id"],
        "pos_id":                pos_id,
        "offset":                offset,
        "model_name":            model_name,
        "patch_mode":            "all_layers_simultaneous",
        "patch_direction":       "corrupt->clean",
        "n_layers_patched":      n_layers,
        "corrupt_patch_pos":     corrupt_patch_pos,
        "corrupt_patch_label":   corrupt_patch_label,
        "clean_patch_pos":       clean_patch_pos,
        "clean_patch_label":     clean_patch_label,
        "sampling_n":            SAMPLING_N,
        "sampling_temp":         SAMPLING_TEMP,
        "max_new_tokens":        MAX_NEW_TOKENS,
        "clean_prompt":          clean_prompt,
        "corrupt_prompt":        corrupt_prompt,
        "clean_rhyme_word":      clean_rhyme_word,
        "corrupt_rhyme_word":    corrupt_rhyme_word,
        "completions":           completions,
        "clean_rhyme_rate":      clean_rate,
        "corrupt_rhyme_rate":    corrupt_rate,
        "delta_corrupt_rate":    delta,
        "baseline": {
            "completions":        baseline_completions,
            "clean_rhyme_rate":   baseline_clean_rate,
            "corrupt_rhyme_rate": baseline_corrupt_rate,
        },
    }

    os.makedirs(pos_dir, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(export, f, indent=2)

    return export

# ── Single Pair ────────────────────────────────────────────────────────────────

def run_pair(model, tokenizer, adapter: ModelAdapter,
             model_name: str, pair: dict, results_dir: str,
             batch_size: int = BATCH_SIZE) -> dict:

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
        print(f"  WARNING: token length mismatch "
              f"({clean_ids.shape[1]} vs {corrupt_ids.shape[1]})")

    pair_dir  = os.path.join(results_dir, pair_id)
    meta_path = os.path.join(pair_dir, "pair_meta.json")

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            pair_meta = json.load(f)
        baseline_completions  = pair_meta.get("baseline_completions", [])
        baseline_clean_rate   = pair_meta["baseline_clean_rate"]
        baseline_corrupt_rate = pair_meta["baseline_corrupt_rate"]
        clean_completion      = pair_meta.get("clean_completion", "")
        corrupt_completion    = pair_meta.get("corrupt_completion", "")
        print(f"  Baseline loaded from disk: "
              f"clean={baseline_clean_rate:.3f} corrupt={baseline_corrupt_rate:.3f}")
    else:
        print("  Greedy baselines...")
        clean_completion   = generate_text(model, tokenizer, adapter, clean_prompt,   0)
        corrupt_completion = generate_text(model, tokenizer, adapter, corrupt_prompt, 0)
        print(f"  Clean   -> {repr(clean_completion[:100])}")
        print(f"  Corrupt -> {repr(corrupt_completion[:100])}")

        print(f"  Unpatched clean baseline (N={SAMPLING_N}, T={SAMPLING_TEMP}, batch={batch_size})...")
        baseline_completions  = sample_completions(
            model, tokenizer, adapter, clean_prompt, SAMPLING_N, SAMPLING_TEMP,
            batch_size=batch_size,
        )
        baseline_clean_rate   = rhyme_rate(baseline_completions, clean_prompt, clean_rhyme_word)
        baseline_corrupt_rate = rhyme_rate(baseline_completions, clean_prompt, corrupt_rhyme_word)
        print(f"    Rhymes with '{clean_rhyme_word}': {baseline_clean_rate:.3f}")
        print(f"    Rhymes with '{corrupt_rhyme_word}': {baseline_corrupt_rate:.3f}")

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
            "baseline_completions":  baseline_completions,
        }
        with open(meta_path, "w") as f:
            json.dump(pair_meta, f, indent=2)

    position_exports = {}
    for pos in POSITIONS:
        export = run_position(
            model, tokenizer, adapter, model_name, pair,
            offset=pos["offset"], pos_id=pos["pos_id"],
            baseline_completions=baseline_completions,
            baseline_clean_rate=baseline_clean_rate,
            baseline_corrupt_rate=baseline_corrupt_rate,
            pair_dir=pair_dir,
            batch_size=batch_size,
        )
        position_exports[pos["pos_id"]] = export

    return position_exports

# ── Aggregate ──────────────────────────────────────────────────────────────────

def compute_aggregate(all_position_exports: list) -> dict:
    """Average corrupt_rhyme_rate across pairs for each position."""
    pos_ids   = [p["pos_id"] for p in POSITIONS]
    aggregate = {}
    for pos_id in pos_ids:
        corrupt_rates = [exports[pos_id]["corrupt_rhyme_rate"]  for exports in all_position_exports]
        clean_rates   = [exports[pos_id]["clean_rhyme_rate"]    for exports in all_position_exports]
        delta_rates   = [exports[pos_id]["delta_corrupt_rate"]  for exports in all_position_exports]
        aggregate[pos_id] = {
            "mean_corrupt_rhyme_rate": sum(corrupt_rates) / len(corrupt_rates),
            "mean_clean_rhyme_rate":   sum(clean_rates)   / len(clean_rates),
            "mean_delta_corrupt_rate": sum(delta_rates)   / len(delta_rates),
            "per_pair_corrupt_rhyme_rate": {
                exports[pos_id]["pair_id"]: exports[pos_id]["corrupt_rhyme_rate"]
                for exports in all_position_exports
            },
        }
    return aggregate

# ── Main ───────────────────────────────────────────────────────────────────────

def model_slug(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name).strip("_")

def run_all(model_name: str, batch_size: int = BATCH_SIZE):
    model, tokenizer, adapter = load_model(model_name)

    slug        = model_slug(model_name)
    result_key  = f"{slug}_all_layers_N{SAMPLING_N}"
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", "ALL_LAYERS_AGGREGATE", result_key,
    )
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results dir: {results_dir}")
    print(f"Batch size:  {batch_size}")
    print(f"Patch mode:  all {adapter.get_n_layers()} layers simultaneously")

    print("\n-- Tokenisation check (positions relative to second newline) --")
    for pair in PROMPT_PAIRS:
        enc = tokenizer(pair["corrupt_prompt"], return_offsets_mapping=True, add_special_tokens=True)
        nl_chars  = [i for i, ch in enumerate(pair["corrupt_prompt"]) if ch == "\n"]
        second_nl = nl_chars[1]
        nl_tok    = next((i for i, (s, e) in enumerate(enc["offset_mapping"]) if s <= second_nl < e), None)
        ids       = enc["input_ids"]
        ctx = " ".join(
            f"i={o}:{repr(tokenizer.decode([ids[nl_tok+o]]))}"
            for o in range(-2, 4) if 0 <= nl_tok+o < len(ids)
        )
        print(f"  {pair['pair_id']}: {ctx}")
    print()

    all_position_exports = []
    for pair in PROMPT_PAIRS:
        position_exports = run_pair(
            model, tokenizer, adapter, model_name, pair, results_dir,
            batch_size=batch_size,
        )
        all_position_exports.append(position_exports)

    agg_path  = os.path.join(results_dir, "aggregate.json")
    aggregate = compute_aggregate(all_position_exports)

    print("\n-- Aggregate Summary (all layers patched simultaneously) --")
    for pos in POSITIONS:
        pos_id = pos["pos_id"]
        r = aggregate[pos_id]
        print(f"  {pos_id}: mean_corrupt_rate={r['mean_corrupt_rhyme_rate']:.3f}  "
              f"mean_delta={r['mean_delta_corrupt_rate']:+.3f}")

    with open(agg_path, "w") as f:
        json.dump({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_name":    model_name,
            "patch_mode":    "all_layers_simultaneous",
            "n_layers":      adapter.get_n_layers(),
            "sampling_n":    SAMPLING_N,
            "sampling_temp": SAMPLING_TEMP,
            "pairs":         [p["pair_id"] for p in PROMPT_PAIRS],
            "positions":     [p["pos_id"]  for p in POSITIONS],
            "aggregate":     aggregate,
        }, f, indent=2)
    print(f"\nAggregate saved to {agg_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified all-layers-simultaneous activation-patching experiment."
    )
    parser.add_argument(
        "--model", type=str,
        help="HuggingFace model ID, e.g. Qwen/Qwen3-8B",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="Print all supported model IDs and exit.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Sequences per generate() call (default: {BATCH_SIZE}). "
             "Reduce to 8-16 for 70B models if you hit OOM.",
    )
    args = parser.parse_args()

    if args.list_models:
        print("Supported models:")
        for m in SUPPORTED_MODELS:
            print(f"  {m}")
        return

    if not args.model:
        parser.error("--model is required (or use --list-models)")

    model_name = args.model
    if model_name not in SUPPORTED_MODELS:
        matches = [m for m in SUPPORTED_MODELS if SUPPORTED_MODELS[m] == model_name]
        if len(matches) == 1:
            model_name = matches[0]
        else:
            print(f"WARNING: '{model_name}' not in SUPPORTED_MODELS list, proceeding anyway.")

    run_all(model_name, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
