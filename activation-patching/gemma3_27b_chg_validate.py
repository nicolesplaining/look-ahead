"""
Validate CHG-facilitating heads against existing top-5 attention-weight heads
on Gemma-3-27B. Also runs random-baseline and attention-matched negative
controls in the same script for a complete comparison plot.

Pipeline:
  1. Load Gemma-3-27B-IT (bf16)
  2. Compute attention weights from i=0 (newline) at each layer/head, for
     the doom_dread corrupt prompt. Use to rank:
       - heads by attention-to-last-word (i=-2)  → "attn_topk" set (paper)
       - heads by attention-to-comma     (i=-1)  → "comma_ctrl" set (negative control)
  3. Load CHG aggregate.json → CHG-facilitating top-5 (composite G+ × G-)
  4. For each head set, run simultaneous patching at i=0 (corrupt → clean)
     across all 5 prompt pairs × N=20 samples; compute mean corrupt rhyme rate.
  5. Random baseline: 10 random draws of 5 heads from the L20-40 region
     (the "candidate" zone), to keep them in the right depth range.

Output: results/gemma3_27b_chg_validate/comparison.json + bar plot.
"""

import json
import os
import random
from contextlib import contextmanager
from collections import defaultdict
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pronouncing
import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.suppress_errors = True
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME      = "google/gemma-3-27b-it"
SAMPLING_N      = 20
SAMPLING_TEMP   = 0.7
MAX_NEW_TOKENS  = 20
BATCH_SIZE      = 20
N_RANDOM_DRAWS  = 10
RANDOM_SEED     = 42
K_HEADS         = 5
CHG_AGG_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "results/gemma3_27b_chg_full/aggregate.json")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results/gemma3_27b_chg_validate")

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

# ── Generation ────────────────────────────────────────────────────────────────

def find_nl_pos(tokenizer, prompt):
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    om  = enc["offset_mapping"]
    nl_chars = [i for i, ch in enumerate(prompt) if ch == "\n"]
    second_nl = nl_chars[1]
    return next(i for i, (s, e) in enumerate(om) if s <= second_nl < e)

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

# ── Cache + patch ─────────────────────────────────────────────────────────────

def cache_head_outputs(model, tokenizer, prompt, nl_pos, layers_needed, device):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    cached = {}
    handles = []
    def make_capture(layer_idx):
        def hook(module, args):
            x = args[0]
            if x.shape[1] > nl_pos:
                cached[layer_idx] = x[0, nl_pos, :].detach().clone()
        return hook
    for layer_idx in layers_needed:
        h = model.model.language_model.layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(
            make_capture(layer_idx))
        handles.append(h)
    try:
        with torch.no_grad():
            model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return cached


@contextmanager
def patch_heads_at_nl(model, corrupt_cache, top_heads, nl_pos, head_dim):
    layer_to_heads = defaultdict(list)
    for layer, head in top_heads:
        layer_to_heads[layer].append(head)
    handles = []
    for layer_idx, heads in layer_to_heads.items():
        corrupt_vec = corrupt_cache[layer_idx]
        def make_hook(corrupt_vec, heads, nl_pos, head_dim):
            def hook(module, args):
                x = args[0].clone()
                if x.shape[1] <= nl_pos:
                    return args
                for h in heads:
                    x[:, nl_pos, h * head_dim:(h + 1) * head_dim] = \
                        corrupt_vec[h * head_dim:(h + 1) * head_dim].to(x.device, dtype=x.dtype)
                return (x,) + args[1:]
            return hook
        h = model.model.language_model.layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(
            make_hook(corrupt_vec, heads, nl_pos, head_dim))
        handles.append(h)
    try:
        yield
    finally:
        for h in handles:
            h.remove()

# ── Compute attention weights from newline position ──────────────────────────

def compute_attention_to_positions(model, tokenizer, prompt, nl_pos, device):
    """Return [n_layers, n_heads, n_positions] attention weights from nl_pos."""
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc, output_attentions=True, use_cache=False)
    # out.attentions: tuple of [B, n_heads, T, T] per layer
    attns = out.attentions
    n_layers = len(attns)
    # Slice attention from nl_pos: shape [n_layers, n_heads, T]
    arr = torch.stack([a[0, :, nl_pos, :] for a in attns], dim=0).float().cpu().numpy()
    return arr  # [L, H, T]

# ── Head set construction ────────────────────────────────────────────────────

def topk_heads_by_attention(attn, target_pos, k):
    """Rank heads by attention from i=0 → target_pos. Return list of (layer, head)."""
    # attn: [L, H, T], we read attn[:, :, target_pos]
    scores = attn[:, :, target_pos]
    flat = [(l, h, scores[l, h])
            for l in range(scores.shape[0]) for h in range(scores.shape[1])]
    flat.sort(key=lambda x: -x[2])
    return [(l, h) for l, h, _ in flat[:k]], flat

def chg_topk_facilitating(agg_path, k, layer_min=0, layer_max=None):
    d = json.load(open(agg_path))
    G_plus  = np.array(d["G_plus_mean"])
    G_minus = np.array(d["G_minus_mean"])
    composite = G_plus * G_minus
    L, H = composite.shape
    if layer_max is None:
        layer_max = L
    flat = [(l, h, composite[l, h])
            for l in range(layer_min, layer_max) for h in range(H)]
    flat.sort(key=lambda x: -x[2])
    return [(l, h) for l, h, _ in flat[:k]], flat

def random_heads(layers, n_heads, k, rng):
    all_heads = [(l, h) for l in layers for h in range(n_heads)]
    return rng.sample(all_heads, k)

# ── Run patching for one head set across all pairs ───────────────────────────

def run_patching_for_set(model, tokenizer, head_set, label, layers_needed,
                          prompt_pairs, n_heads, head_dim, device):
    """Returns dict: pair_id -> corrupt_rhyme_rate."""
    results = {}
    for pair in prompt_pairs:
        clean_prompt   = pair["clean_prompt"]
        corrupt_prompt = pair["corrupt_prompt"]
        corrupt_rw     = pair["corrupt_rhyme_word"]

        nl_clean   = find_nl_pos(tokenizer, clean_prompt)
        nl_corrupt = find_nl_pos(tokenizer, corrupt_prompt)

        cache = cache_head_outputs(model, tokenizer, corrupt_prompt, nl_corrupt,
                                    layers_needed, device)
        with patch_heads_at_nl(model, cache, head_set, nl_clean, head_dim):
            comps = sample_completions(model, tokenizer, clean_prompt,
                                        SAMPLING_N, SAMPLING_TEMP, device)
        cr = rhyme_rate(comps, clean_prompt, corrupt_rw)
        results[pair["pair_id"]] = cr
        print(f"    [{label}] {pair['pair_id']}: corrupt_rhyme_rate={cr:.3f}", flush=True)
    return results

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {MODEL_NAME} (bf16)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # eager attn so output_attentions=True works for the head-ranking pass
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    cfg = model.config.text_config
    n_heads  = cfg.num_attention_heads
    head_dim = cfg.head_dim if hasattr(cfg, "head_dim") else cfg.hidden_size // n_heads
    n_layers = cfg.num_hidden_layers
    device   = model.model.language_model.embed_tokens.weight.device
    print(f"Loaded. L={n_layers} H={n_heads} d_head={head_dim}", flush=True)

    # ── Compute attention weights on doom_dread corrupt prompt ──
    p0 = PROMPT_PAIRS[0]  # doom_dread
    nl_pos = find_nl_pos(tokenizer, p0["corrupt_prompt"])
    enc = tokenizer(p0["corrupt_prompt"], add_special_tokens=True)
    n_tok = len(enc["input_ids"])
    last_word_pos = nl_pos - 2  # i=-2 in Gemma (last word, before comma)
    comma_pos     = nl_pos - 1  # i=-1 (comma)
    print(f"\nPositions on doom_dread/corrupt: nl={nl_pos} comma={comma_pos} last_word={last_word_pos}")

    print("Computing attention weights from newline...")
    attn = compute_attention_to_positions(model, tokenizer, p0["corrupt_prompt"], nl_pos, device)
    print(f"  attn shape={attn.shape}")

    # Head sets
    # 1. Paper's hardcoded top-5 (from gemma3_27b_topk_head_patching.py).
    # These are derived from cross-prompt averaging; using them ensures direct
    # comparability with the paper's reported ~0.73 result.
    attn_top5 = [(30, 4), (28, 14), (28, 15), (30, 5), (28, 29)]
    attn_flat = []  # not used downstream
    # 2. attn-to-comma top-K (negative control)
    comma_top5, comma_flat = topk_heads_by_attention(attn, comma_pos, K_HEADS)
    # 3. CHG facilitating top-K (composite)
    chg_top5,  chg_flat = chg_topk_facilitating(CHG_AGG_PATH, K_HEADS)
    chg_top10, _        = chg_topk_facilitating(CHG_AGG_PATH, 10)

    # 4. Random sets — drawn from same depth band as attn_top5 (median ± window)
    layer_band = sorted({l for l, _ in attn_top5 + chg_top5})
    band_min, band_max = max(0, min(layer_band) - 2), min(n_layers, max(layer_band) + 3)
    random_layers = list(range(band_min, band_max))
    rng = random.Random(RANDOM_SEED)
    random_sets = [random_heads(random_layers, n_heads, K_HEADS, rng)
                   for _ in range(N_RANDOM_DRAWS)]
    print(f"\nRandom-baseline layer band: {band_min}-{band_max-1}", flush=True)

    # Print head sets
    def fmt_heads(hs):
        return ", ".join(f"L{l}H{h}" for l, h in hs)
    print(f"\nHead sets (k={K_HEADS}):")
    print(f"  attn_topk   : {fmt_heads(attn_top5)}")
    print(f"  comma_ctrl  : {fmt_heads(comma_top5)}")
    print(f"  chg_topk    : {fmt_heads(chg_top5)}")
    print(f"  chg_top10   : {fmt_heads(chg_top10)}")

    # Layers needed for caching
    all_heads = (attn_top5 + comma_top5 + chg_top5 + chg_top10
                 + [h for s in random_sets for h in s])
    layers_needed = sorted({l for l, _ in all_heads})

    all_results = {}

    print(f"\n{'='*60}\nattn_topk  (paper's top-{K_HEADS} attention-weight)\n{'='*60}")
    all_results["attn_topk"] = run_patching_for_set(
        model, tokenizer, attn_top5, "attn_topk", layers_needed,
        PROMPT_PAIRS, n_heads, head_dim, device)

    print(f"\n{'='*60}\ncomma_ctrl (attn-to-comma, negative control)\n{'='*60}")
    all_results["comma_ctrl"] = run_patching_for_set(
        model, tokenizer, comma_top5, "comma_ctrl", layers_needed,
        PROMPT_PAIRS, n_heads, head_dim, device)

    print(f"\n{'='*60}\nchg_topk   (CHG facilitating top-{K_HEADS})\n{'='*60}")
    all_results["chg_topk"] = run_patching_for_set(
        model, tokenizer, chg_top5, "chg_topk", layers_needed,
        PROMPT_PAIRS, n_heads, head_dim, device)

    print(f"\n{'='*60}\nchg_top10  (CHG facilitating top-10)\n{'='*60}")
    all_results["chg_top10"] = run_patching_for_set(
        model, tokenizer, chg_top10, "chg_top10", layers_needed,
        PROMPT_PAIRS, n_heads, head_dim, device)

    print(f"\n{'='*60}\nrandom_baseline (10 draws of {K_HEADS})\n{'='*60}")
    all_results["random_baseline"] = []
    for i, rset in enumerate(random_sets):
        print(f"  draw {i+1}/{N_RANDOM_DRAWS}: {fmt_heads(rset)}")
        r = run_patching_for_set(model, tokenizer, rset, f"random_{i}",
                                  layers_needed, PROMPT_PAIRS, n_heads, head_dim, device)
        all_results["random_baseline"].append(r)

    # ── Aggregate ──
    summary = {}
    for label, res in all_results.items():
        if label == "random_baseline":
            means = [np.mean(list(r.values())) for r in res]
            summary[label] = {
                "mean_corrupt_rhyme_rate":  float(np.mean(means)),
                "std_corrupt_rhyme_rate":   float(np.std(means)),
                "per_draw_means": means,
            }
        else:
            summary[label] = {
                "mean_corrupt_rhyme_rate": float(np.mean(list(res.values()))),
                "per_pair": res,
            }

    out = {
        "timestamp_utc":     datetime.now(timezone.utc).isoformat(),
        "model_name":        MODEL_NAME,
        "k":                 K_HEADS,
        "n_random_draws":    N_RANDOM_DRAWS,
        "head_sets": {
            "attn_topk":   attn_top5,
            "comma_ctrl":  comma_top5,
            "chg_topk":    chg_top5,
            "chg_top10":   chg_top10,
            "random_sets": random_sets,
        },
        "summary":           summary,
        "all_results":       all_results,
    }
    with open(os.path.join(OUT_DIR, "comparison.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ── Plot ──
    labels = ["attn_topk\n(paper's 5)", "comma_ctrl\n(neg ctrl)",
              "chg_topk\n(CHG 5)", "chg_top10\n(CHG 10)",
              f"random×{N_RANDOM_DRAWS}\n(L{band_min}-{band_max-1})"]
    means = [
        summary["attn_topk"]["mean_corrupt_rhyme_rate"],
        summary["comma_ctrl"]["mean_corrupt_rhyme_rate"],
        summary["chg_topk"]["mean_corrupt_rhyme_rate"],
        summary["chg_top10"]["mean_corrupt_rhyme_rate"],
        summary["random_baseline"]["mean_corrupt_rhyme_rate"],
    ]
    stds = [None, None, None, None, summary["random_baseline"]["std_corrupt_rhyme_rate"]]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    colors = ["#2255cc", "#999999", "#22aa44", "#11883a", "#cc6677"]
    for i, (lab, m, s, c) in enumerate(zip(labels, means, stds, colors)):
        if s is not None:
            ax.bar(i, m, yerr=s, color=c, capsize=5)
        else:
            ax.bar(i, m, color=c)
        ax.text(i, m + 0.02, f"{m:.3f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Corrupt rhyme rate (mean over 5 pairs × N=20)", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title("Gemma-3-27B simultaneous-patching head-set comparison", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "head_set_comparison.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"\nSaved {out_png}")

    print("\n=== Final summary ===")
    for lab, m, s in zip(labels, means, stds):
        if s is not None:
            print(f"  {lab.replace(chr(10), ' '):<35}  mean={m:.3f} ± {s:.3f}")
        else:
            print(f"  {lab.replace(chr(10), ' '):<35}  mean={m:.3f}")


if __name__ == "__main__":
    main()
