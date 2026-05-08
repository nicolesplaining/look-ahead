"""
TWO-STAGE path-patching K-sweep on Gemma-3-27B-IT (Wang et al. 2022 IOI style).

Differs from the single-stage `gemma3_27b_path_patch_sweep.py` only in stage 1:

  Single-stage stage 1:
    forward(corrupt_prompt) -> cache heads' o_proj input at i=0

  Two-stage stage 1 (this script):
    1.a  cache corrupt's residual stream at i=-2 at each layer's input
    1.b  forward(clean_prompt) BUT: at each layer's input, replace the residual
         at i=-2 with corrupt's cached i=-2 residual ("freeze patching"). While
         this surgical-upstream forward runs, capture the candidate heads'
         o_proj input at i=0.

    The captured head outputs at i=0 now reflect ONLY information that was
    routed through i=-2 (everything else in the prompt is clean).

  Stage 2: identical to single-stage. Forward clean_prompt; substitute selected
           heads' o_proj-input slices at i=0 with stage-1 cached values; sample.

Output dir: results/gemma3_27b_path_patch_two_stage/worker{0,1}/
Same K-sweep (1, 3, 5, 10, 15) and same controls (attnweight, commactrl, random×5).

Usage:
    python3 gemma3_27b_path_patch_two_stage.py --worker_id 0 --num_workers 2
"""

import argparse
import gc
import json
import os
import random
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pronouncing
import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.suppress_errors = True

from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

# ── Config (mirrors single-stage script) ─────────────────────────────────────

MODEL_NAME      = "google/gemma-3-27b-it"
SAMPLING_N      = 20
SAMPLING_TEMP   = 0.7
MAX_NEW_TOKENS  = 20
BATCH_SIZE      = 20
K_VALUES        = [1, 3, 5, 6, 7, 8, 9, 10, 15]
N_RANDOM_DRAWS  = 5

PROMPT_PAIRS = [
    {"pair_id": "doom_dread",
     "clean_prompt":   "A rhyming couplet:\nThe empty house was filled with silent doom,\nwhen suddenly they",
     "corrupt_prompt": "A rhyming couplet:\nThe empty house was filled with silent dread,\nwhen suddenly they",
     "clean_rhyme_word": "doom", "corrupt_rhyme_word": "dread"},
    {"pair_id": "bliss_joy",
     "clean_prompt":   "A rhyming couplet:\nThe children laughed in bliss,\nuntil they all",
     "corrupt_prompt": "A rhyming couplet:\nThe children laughed in joy,\nuntil they all",
     "clean_rhyme_word": "bliss", "corrupt_rhyme_word": "joy"},
    {"pair_id": "dark_night",
     "clean_prompt":   "A rhyming couplet:\nShe wandered home alone into the dark,\nand then she",
     "corrupt_prompt": "A rhyming couplet:\nShe wandered home alone into the night,\nand then she",
     "clean_rhyme_word": "dark", "corrupt_rhyme_word": "night"},
    {"pair_id": "grief_pain",
     "clean_prompt":   "A rhyming couplet:\nI never knew the depth of such grief,\nas though the",
     "corrupt_prompt": "A rhyming couplet:\nI never knew the depth of such pain,\nas though the",
     "clean_rhyme_word": "grief", "corrupt_rhyme_word": "pain"},
    {"pair_id": "fright_fear",
     "clean_prompt":   "A rhyming couplet:\nShe felt a sudden sense of fright,\nand hoped that",
     "corrupt_prompt": "A rhyming couplet:\nShe felt a sudden sense of fear,\nand hoped that",
     "clean_rhyme_word": "fright", "corrupt_rhyme_word": "fear"},
]

PAPER_TOP_HEADS = [
    (30, 4), (28, 14), (28, 15), (30, 5), (28, 29),
    (34, 24), (33, 14), (34, 14), (36, 21), (31, 24),
]

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "results/gemma3_27b_path_patch_two_stage")

# ── Rhyme helpers ────────────────────────────────────────────────────────────

def _rhyme_score(w1, w2):
    p1 = pronouncing.phones_for_word(w1.lower().strip())
    p2 = pronouncing.phones_for_word(w2.lower().strip())
    if not p1 or not p2:
        return None
    rp1 = pronouncing.rhyming_part(p1[0])
    rp2 = pronouncing.rhyming_part(p2[0])
    return (rp1 == rp2) if (rp1 and rp2) else None

def last_word(text):
    for w in reversed(text.split()):
        cleaned = w.strip(".,!?\"'—;: ")
        if cleaned.isalpha():
            return cleaned.lower()
    return ""

def word_before_nth_newline(text, n):
    if n <= 0:
        return ""
    nls = [i for i, ch in enumerate(text) if ch == "\n"]
    if len(nls) < n:
        return ""
    end   = nls[n - 1]
    start = nls[n - 2] + 1 if n >= 2 else 0
    return last_word(text[start:end])

def extract_rhyme_word(full_text, prompt):
    idx = prompt.count("\n") + 1
    w = word_before_nth_newline(full_text, idx)
    if w:
        return w
    if full_text.startswith(prompt):
        return last_word(full_text[len(prompt):])
    return last_word(full_text)

def rhyme_rate(completions, prompt, rhyme_word):
    hits = sum(1 for c in completions
               if _rhyme_score(extract_rhyme_word(c, prompt), rhyme_word) is True)
    return hits / len(completions) if completions else 0.0

def find_nl_pos(tokenizer, prompt):
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    om = enc["offset_mapping"]
    nl_chars = [i for i, ch in enumerate(prompt) if ch == "\n"]
    second_nl = nl_chars[1]
    return next(i for i, (s, e) in enumerate(om) if s <= second_nl < e)

# ── Generation ────────────────────────────────────────────────────────────────

def sample_completions(model, tokenizer, prompt, n, temperature, device):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    completions = []
    remaining = n
    while remaining > 0:
        b = min(remaining, BATCH_SIZE)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True, temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=b,
            )
        for row in out:
            completions.append(tokenizer.decode(row, skip_special_tokens=True))
        remaining -= b
    return completions

# ── Two-stage primitives ──────────────────────────────────────────────────────

def cache_residual_at_pos(model, tokenizer, prompt, target_pos, device):
    """Cache the residual stream at `target_pos` at each layer's INPUT.

    Returns dict: layer_idx -> Tensor[d_model] at target_pos.
    """
    layers = model.model.language_model.layers
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    cache = {}
    handles = []
    def make_capture(idx):
        def hook(module, args, kwargs):
            # Layer module pre-hook: input is (hidden_states, ...) or kwargs.
            if args:
                h = args[0]
            else:
                h = kwargs.get("hidden_states")
            if h is not None and h.dim() == 3 and h.shape[1] > target_pos:
                cache[idx] = h[0, target_pos, :].detach().clone()
        return hook
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(make_capture(i), with_kwargs=True))
    try:
        with torch.no_grad():
            model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    if any(i not in cache for i in range(len(layers))):
        missing = [i for i in range(len(layers)) if i not in cache]
        raise RuntimeError(f"Failed to cache residual at layers: {missing}")
    return cache


def stage1_capture_head_outputs(model, tokenizer, clean_prompt,
                                  corrupt_residual_cache, target_resid_pos,
                                  layers_to_capture, nl_pos, device):
    """Stage 1 of two-stage path patching.

    Forward clean_prompt with the residual stream at target_resid_pos REPLACED
    with corrupt's cached residual at each layer's input. While this forward
    runs, capture the o_proj input at nl_pos for layers in layers_to_capture.

    Returns dict: layer_idx -> Tensor[H*D] at nl_pos.
    """
    layers = model.model.language_model.layers
    enc = tokenizer(clean_prompt, return_tensors="pt").to(device)
    head_cache = {}
    handles = []

    # (a) per-layer pre-hooks to swap residual at target_resid_pos
    def make_resid_swap(idx):
        corrupt_vec = corrupt_residual_cache[idx]  # [d_model]
        def hook(module, args, kwargs):
            if args:
                h = args[0]
                use_args = True
            else:
                h = kwargs.get("hidden_states")
                use_args = False
            if h is None or h.dim() != 3 or h.shape[1] <= target_resid_pos:
                return
            h_new = h.clone()
            h_new[:, target_resid_pos, :] = corrupt_vec.to(h_new.device, dtype=h_new.dtype)
            if use_args:
                return ((h_new,) + args[1:], kwargs)
            else:
                kwargs["hidden_states"] = h_new
                return (args, kwargs)
        return hook

    # (b) o_proj pre-hooks to capture head outputs at nl_pos
    def make_oproj_capture(idx):
        def hook(module, args):
            x = args[0]
            if x.shape[1] > nl_pos:
                head_cache[idx] = x[0, nl_pos, :].detach().clone()
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(make_resid_swap(i), with_kwargs=True))
        if i in layers_to_capture:
            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(make_oproj_capture(i)))

    try:
        with torch.no_grad():
            model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    return head_cache


@contextmanager
def patch_heads_at_position(model, corrupt_cache, head_set, target_pos, head_dim):
    """Stage 2 (unchanged from single-stage script)."""
    layer_to_heads = defaultdict(list)
    for layer, head in head_set:
        layer_to_heads[layer].append(head)
    handles = []
    for layer_idx, heads in layer_to_heads.items():
        if layer_idx not in corrupt_cache:
            continue  # skip if cache missing (shouldn't happen)
        corrupt_vec = corrupt_cache[layer_idx]
        def make_hook(corrupt_vec, heads, target_pos, head_dim):
            def hook(module, args):
                x = args[0].clone()
                if x.shape[1] <= target_pos:
                    return args
                for h in heads:
                    x[:, target_pos, h * head_dim:(h + 1) * head_dim] = \
                        corrupt_vec[h * head_dim:(h + 1) * head_dim].to(x.device, dtype=x.dtype)
                return (x,) + args[1:]
            return hook
        h = model.model.language_model.layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(
            make_hook(corrupt_vec, heads, target_pos, head_dim))
        handles.append(h)
    try:
        yield
    finally:
        for h in handles:
            h.remove()

# ── Cell key + execution ─────────────────────────────────────────────────────

def cell_key(K, ranking, seed, pair_id):
    seed_str = "" if seed is None else f"_seed{seed}"
    return f"K{K:02d}_{ranking}{seed_str}_pair_{pair_id}"

def cell_path(K, ranking, seed, pair_id, out_dir):
    return os.path.join(out_dir, cell_key(K, ranking, seed, pair_id) + ".json")


def run_cell(model, tokenizer, K, ranking, seed, pair, head_set,
              head_dim, device, out_dir, last_word_pos_for_pair):
    key = cell_key(K, ranking, seed, pair["pair_id"])
    path = cell_path(K, ranking, seed, pair["pair_id"], out_dir)
    if os.path.exists(path):
        return json.load(open(path))

    clean_prompt   = pair["clean_prompt"]
    corrupt_prompt = pair["corrupt_prompt"]
    corrupt_rw     = pair["corrupt_rhyme_word"]
    clean_rw       = pair["clean_rhyme_word"]

    nl_clean = find_nl_pos(tokenizer, clean_prompt)
    last_word_pos = last_word_pos_for_pair  # i=-2 in Gemma; precomputed

    # Stage 1.a: cache corrupt's residual at i=-2 at each layer
    corrupt_resid = cache_residual_at_pos(
        model, tokenizer, corrupt_prompt, last_word_pos, device,
    )

    # Stage 1.b: clean forward with i=-2 residual frozen to corrupt's;
    #            capture selected heads' o_proj input at i=0
    layers_needed = sorted({l for l, _ in head_set})
    head_cache = stage1_capture_head_outputs(
        model, tokenizer, clean_prompt,
        corrupt_resid, last_word_pos, layers_needed, nl_clean, device,
    )

    # Stage 2: clean forward with selected heads' o_proj input at i=0 swapped
    with patch_heads_at_position(model, head_cache, head_set, nl_clean, head_dim):
        comps = sample_completions(model, tokenizer, clean_prompt,
                                    SAMPLING_N, SAMPLING_TEMP, device)
    cr  = rhyme_rate(comps, clean_prompt, corrupt_rw)
    clr = rhyme_rate(comps, clean_prompt, clean_rw)

    result = {
        "key": key, "K": K, "ranking": ranking, "seed": seed,
        "pair_id": pair["pair_id"],
        "head_set": head_set,
        "corrupt_rhyme_rate": cr,
        "clean_rhyme_rate":   clr,
        "completions":        comps,
        "method": "two_stage_path_patching",
        "timestamp_utc":      datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return result

# ── Head-set construction ────────────────────────────────────────────────────

def compute_attention_to_positions(model, tokenizer, prompt, nl_pos, device):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc, output_attentions=True, use_cache=False)
    arr = torch.stack([a[0, :, nl_pos, :] for a in out.attentions], dim=0).float().cpu().numpy()
    return arr

def topk_heads_by_attention(attn, target_pos, k):
    scores = attn[:, :, target_pos]
    flat = [(l, h, scores[l, h])
            for l in range(scores.shape[0]) for h in range(scores.shape[1])]
    flat.sort(key=lambda x: -x[2])
    return [(l, h) for l, h, _ in flat[:k]]

def random_heads(layers_band, n_heads, k, rng):
    pool = [(l, h) for l in layers_band for h in range(n_heads)]
    return rng.sample(pool, k)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_id",   type=int, required=True)
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--out_dir",     type=str, default=None)
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(OUT_ROOT, f"worker{args.worker_id}")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[worker {args.worker_id}/{args.num_workers}]  out_dir={args.out_dir}", flush=True)

    print(f"Loading {MODEL_NAME} (bf16, eager attn)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()
    cfg = model.config.text_config
    n_layers = cfg.num_hidden_layers
    n_heads  = cfg.num_attention_heads
    head_dim = cfg.head_dim if hasattr(cfg, "head_dim") else cfg.hidden_size // n_heads
    device = model.model.language_model.embed_tokens.weight.device
    print(f"Loaded. L={n_layers} H={n_heads} d_head={head_dim}", flush=True)

    # ── Build attention-weight ranking on doom_dread/corrupt ──
    p0 = PROMPT_PAIRS[0]
    nl_pos = find_nl_pos(tokenizer, p0["corrupt_prompt"])
    last_word_pos = nl_pos - 2  # Gemma: last word at i=-2
    print("Computing attention weights for ranking...", flush=True)
    attn = compute_attention_to_positions(model, tokenizer,
                                           p0["corrupt_prompt"], nl_pos, device)
    attnweight_topN = topk_heads_by_attention(attn, last_word_pos, 30)
    attnweight_topN = PAPER_TOP_HEADS + [
        h for h in attnweight_topN if h not in PAPER_TOP_HEADS
    ][:20]
    print(f"  Top-30 attn-weight heads: {attnweight_topN[:10]} ...", flush=True)

    comma_pos = nl_pos - 1
    comma_topN = topk_heads_by_attention(attn, comma_pos, 30)
    print(f"  Top-10 comma heads: {comma_topN[:10]}", flush=True)

    band_layers = sorted({l for l, _ in attnweight_topN[:15]})
    band_min, band_max = max(0, min(band_layers) - 2), min(n_layers, max(band_layers) + 3)
    random_layer_band = list(range(band_min, band_max))
    print(f"  Random layer band: {band_min}-{band_max-1}", flush=True)

    # Precompute last_word_pos per pair (clean prompt's i=-2)
    last_word_pos_per_pair = {}
    for pair in PROMPT_PAIRS:
        nl_p = find_nl_pos(tokenizer, pair["clean_prompt"])
        last_word_pos_per_pair[pair["pair_id"]] = nl_p - 2

    # ── Build cell list ──
    cells = []
    for K in K_VALUES:
        cells.append({"K": K, "ranking": "attnweight", "seed": None,
                       "head_set": attnweight_topN[:K]})
        cells.append({"K": K, "ranking": "commactrl", "seed": None,
                       "head_set": comma_topN[:K]})
        for seed in range(N_RANDOM_DRAWS):
            rng = random.Random(seed * 7919 + K)
            cells.append({"K": K, "ranking": "random", "seed": seed,
                           "head_set": random_heads(random_layer_band, n_heads, K, rng)})

    full_cells = []
    for c in cells:
        for pair in PROMPT_PAIRS:
            full_cells.append({**c, "pair": pair})

    full_cells.sort(key=lambda c: (c["K"], c["ranking"], c["seed"] or -1, c["pair"]["pair_id"]))
    print(f"\n  Total cells: {len(full_cells)}", flush=True)

    my_cells = [c for i, c in enumerate(full_cells)
                if i % args.num_workers == args.worker_id]
    print(f"  Worker {args.worker_id} cells: {len(my_cells)}", flush=True)

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "worker_id":   args.worker_id,
            "num_workers": args.num_workers,
            "n_cells":     len(my_cells),
            "method":      "two_stage_path_patching",
            "config": {
                "k_values":        K_VALUES,
                "n_random_draws":  N_RANDOM_DRAWS,
                "sampling_n":      SAMPLING_N,
                "sampling_temp":   SAMPLING_TEMP,
                "max_new_tokens":  MAX_NEW_TOKENS,
                "attnweight_topN": attnweight_topN,
                "comma_topN":      comma_topN,
                "random_layer_band": [band_min, band_max - 1],
            },
        }, f, indent=2)

    t0 = time.time()
    for i, cell in enumerate(my_cells):
        K        = cell["K"]
        ranking  = cell["ranking"]
        seed     = cell["seed"]
        pair     = cell["pair"]
        head_set = cell["head_set"]
        last_word_pos_pair = last_word_pos_per_pair[pair["pair_id"]]

        key = cell_key(K, ranking, seed, pair["pair_id"])
        path = cell_path(K, ranking, seed, pair["pair_id"], args.out_dir)
        if os.path.exists(path):
            print(f"  [{i+1}/{len(my_cells)}] {key} (cached)", flush=True)
            continue

        cell_t0 = time.time()
        try:
            r = run_cell(model, tokenizer, K, ranking, seed, pair, head_set,
                          head_dim, device, args.out_dir, last_word_pos_pair)
            elapsed = time.time() - cell_t0
            total_elapsed = time.time() - t0
            eta = total_elapsed / (i + 1) * (len(my_cells) - i - 1)
            print(f"  [{i+1}/{len(my_cells)}] {key}  cr={r['corrupt_rhyme_rate']:.3f}  "
                  f"t={elapsed:.0f}s  total={total_elapsed:.0f}s  eta={eta:.0f}s",
                  flush=True)
        except Exception as e:
            print(f"  [{i+1}/{len(my_cells)}] {key}  FAILED: {type(e).__name__}: {e}",
                  flush=True)
            import traceback
            traceback.print_exc()

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n[worker {args.worker_id}] DONE  total={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
