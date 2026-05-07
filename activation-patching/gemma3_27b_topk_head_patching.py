"""
Patch top-k attention heads simultaneously at the newline position (i=0)
for Gemma-3-27B, corrupt -> clean direction.

Top heads are identified by attention weight from i=0 (newline) -> i=-2 (rhyme word).
We patch the head outputs before the o_proj by hooking into o_proj's input.

Tries k = 1, 2, 3, 5, 10 heads and reports aggregate corrupt rhyme rate.

Usage:
    python gemma3_27b_topk_head_patching.py
"""

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import pronouncing
from tqdm import tqdm
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME     = "google/gemma-3-27b-it"
SAMPLING_N     = 20
SAMPLING_TEMP  = 0.7
MAX_NEW_TOKENS = 20
BATCH_SIZE     = 20

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "gemma3_27b_topk_head_patching")

# Top heads from attention pattern analysis (layer, head), ordered by attn weight
TOP_HEADS_RANKED = [
    (30, 4),   # 0.9891
    (28, 14),  # 0.9727
    (28, 15),  # 0.9539
    (30, 5),   # 0.5543
    (28, 29),  # 0.5137
    (34, 24),  # 0.4902
    (33, 14),  # 0.4840
    (34, 14),  # 0.4492
    (36, 21),  # 0.3613
    (31, 24),  # 0.3512
]

K_VALUES = [1, 2, 3, 5, 10]

PROMPT_PAIRS = [
    {
        "pair_id":        "doom_dread",
        "clean_prompt":   "A rhyming couplet:\nThe empty house was filled with silent doom,\nwhen suddenly they",
        "corrupt_prompt": "A rhyming couplet:\nThe empty house was filled with silent dread,\nwhen suddenly they",
        "clean_rhyme_word":   "doom",
        "corrupt_rhyme_word": "dread",
    },
    {
        "pair_id":        "bliss_joy",
        "clean_prompt":   "A rhyming couplet:\nThe children laughed in bliss,\nuntil they all",
        "corrupt_prompt": "A rhyming couplet:\nThe children laughed in joy,\nuntil they all",
        "clean_rhyme_word":   "bliss",
        "corrupt_rhyme_word": "joy",
    },
    {
        "pair_id":        "dark_night",
        "clean_prompt":   "A rhyming couplet:\nShe wandered home alone into the dark,\nand then she",
        "corrupt_prompt": "A rhyming couplet:\nShe wandered home alone into the night,\nand then she",
        "clean_rhyme_word":   "dark",
        "corrupt_rhyme_word": "night",
    },
    {
        "pair_id":        "grief_pain",
        "clean_prompt":   "A rhyming couplet:\nI never knew the depth of such grief,\nas though the",
        "corrupt_prompt": "A rhyming couplet:\nI never knew the depth of such pain,\nas though the",
        "clean_rhyme_word":   "grief",
        "corrupt_rhyme_word": "pain",
    },
    {
        "pair_id":        "fright_fear",
        "clean_prompt":   "A rhyming couplet:\nShe felt a sudden sense of fright,\nand hoped that",
        "corrupt_prompt": "A rhyming couplet:\nShe felt a sudden sense of fear,\nand hoped that",
        "clean_rhyme_word":   "fright",
        "corrupt_rhyme_word": "fear",
    },
]

# ── Rhyme helpers ─────────────────────────────────────────────────────────────

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
        c = w.strip(".,!?\"'—;: ")
        if c.isalpha():
            return c.lower()
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

# ── Position finding ──────────────────────────────────────────────────────────

def find_nl_pos(tokenizer, prompt):
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    om  = enc["offset_mapping"]
    nl_chars = [i for i, ch in enumerate(prompt) if ch == "\n"]
    second_nl = nl_chars[1]
    return next(i for i, (s, e) in enumerate(om) if s <= second_nl < e)

# ── Generation ────────────────────────────────────────────────────────────────

def sample_completions(model, tokenizer, prompt, n, temperature, device):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    completions = []
    remaining = n
    while remaining > 0:
        this_batch = min(remaining, BATCH_SIZE)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=this_batch,
            )
        for row in out:
            completions.append(tokenizer.decode(row, skip_special_tokens=True))
        remaining -= this_batch
    return completions

# ── Head output caching and patching ─────────────────────────────────────────

def cache_head_outputs(model, tokenizer, prompt, nl_pos, layers_needed, device):
    """
    Cache the input to o_proj (= concatenated head outputs) at nl_pos,
    for each layer in layers_needed.
    Returns dict: layer -> Tensor[n_heads * head_dim] at nl_pos.
    """
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    cached = {}
    handles = []

    def make_capture(layer_idx):
        def hook(module, args):
            # args[0]: [batch, seq, n_heads * head_dim]
            x = args[0]
            if x.shape[1] > nl_pos:
                cached[layer_idx] = x[0, nl_pos, :].detach().clone()
        return hook

    for layer_idx in layers_needed:
        h = model.model.language_model.layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(
            make_capture(layer_idx)
        )
        handles.append(h)

    try:
        with torch.no_grad():
            model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    return cached


@contextmanager
def patch_heads_at_nl(model, corrupt_cache, top_heads, nl_pos, n_heads, head_dim):
    """
    During a clean forward pass, replace specific head slices at nl_pos
    with the cached corrupt values (input to o_proj level).
    top_heads: list of (layer, head) pairs to patch simultaneously.
    """
    # Group by layer
    from collections import defaultdict
    layer_to_heads = defaultdict(list)
    for layer, head in top_heads:
        layer_to_heads[layer].append(head)

    handles = []
    for layer_idx, heads in layer_to_heads.items():
        corrupt_vec = corrupt_cache[layer_idx]  # [n_heads * head_dim]

        def make_hook(corrupt_vec, heads, nl_pos, head_dim):
            def hook(module, args):
                x = args[0].clone()  # [batch, seq, n_heads * head_dim]
                if x.shape[1] <= nl_pos:
                    return args
                for h in heads:
                    x[:, nl_pos, h * head_dim:(h + 1) * head_dim] = \
                        corrupt_vec[h * head_dim:(h + 1) * head_dim].to(x.device, dtype=x.dtype)
                return (x,) + args[1:]
            return hook

        h = model.model.language_model.layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(
            make_hook(corrupt_vec, heads, nl_pos, head_dim)
        )
        handles.append(h)

    try:
        yield
    finally:
        for h in handles:
            h.remove()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    n_heads  = model.config.text_config.num_attention_heads
    head_dim = model.config.text_config.head_dim
    device   = model.model.language_model.embed_tokens.weight.device
    print(f"Loaded. Heads={n_heads}, head_dim={head_dim}")

    layers_needed = sorted(set(layer for layer, _ in TOP_HEADS_RANKED))

    all_results = []

    for pair in PROMPT_PAIRS:
        pair_id        = pair["pair_id"]
        clean_prompt   = pair["clean_prompt"]
        corrupt_prompt = pair["corrupt_prompt"]
        clean_rw       = pair["clean_rhyme_word"]
        corrupt_rw     = pair["corrupt_rhyme_word"]

        print(f"\n{'='*60}")
        print(f"Pair: {pair_id}  ({clean_rw!r} vs {corrupt_rw!r})")
        print(f"{'='*60}")

        nl_pos_clean   = find_nl_pos(tokenizer, clean_prompt)
        nl_pos_corrupt = find_nl_pos(tokenizer, corrupt_prompt)
        print(f"  nl_pos: clean={nl_pos_clean}, corrupt={nl_pos_corrupt}")

        # Baseline (unpatched clean)
        print(f"  Baseline (N={SAMPLING_N})...")
        baseline = sample_completions(model, tokenizer, clean_prompt,
                                      SAMPLING_N, SAMPLING_TEMP, device)
        bl_clean_rate   = rhyme_rate(baseline, clean_prompt, clean_rw)
        bl_corrupt_rate = rhyme_rate(baseline, clean_prompt, corrupt_rw)
        print(f"    clean_rate={bl_clean_rate:.3f}  corrupt_rate={bl_corrupt_rate:.3f}")

        # Cache corrupt head outputs at nl_pos
        print(f"  Caching corrupt head outputs...")
        corrupt_cache = cache_head_outputs(
            model, tokenizer, corrupt_prompt, nl_pos_corrupt,
            layers_needed, device
        )

        pair_results = {
            "pair_id": pair_id,
            "baseline_clean_rate":   bl_clean_rate,
            "baseline_corrupt_rate": bl_corrupt_rate,
            "k_results": [],
        }

        for k in K_VALUES:
            top_k = TOP_HEADS_RANKED[:k]
            head_label = ", ".join(f"L{l}H{h}" for l, h in top_k)
            print(f"\n  k={k} heads: {head_label}")

            with patch_heads_at_nl(model, corrupt_cache, top_k,
                                   nl_pos_clean, n_heads, head_dim):
                completions = sample_completions(model, tokenizer, clean_prompt,
                                                 SAMPLING_N, SAMPLING_TEMP, device)

            cr = rhyme_rate(completions, clean_prompt, corrupt_rw)
            clr = rhyme_rate(completions, clean_prompt, clean_rw)
            delta = cr - bl_corrupt_rate
            print(f"    corrupt_rate={cr:.3f}  clean_rate={clr:.3f}  delta={delta:+.3f}")

            pair_results["k_results"].append({
                "k":              k,
                "heads":          top_k,
                "corrupt_rate":   cr,
                "clean_rate":     clr,
                "delta":          delta,
                "completions":    completions,
            })

        all_results.append(pair_results)

    # ── Aggregate across pairs ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("AGGREGATE SUMMARY")
    print(f"{'='*60}")

    agg_by_k = {}
    for k in K_VALUES:
        corrupt_rates = [
            next(r for r in pr["k_results"] if r["k"] == k)["corrupt_rate"]
            for pr in all_results
        ]
        agg_by_k[k] = sum(corrupt_rates) / len(corrupt_rates)
        print(f"  k={k:2d}: mean_corrupt_rate={agg_by_k[k]:.3f}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_json = os.path.join(OUT_DIR, "results.json")
    with open(out_json, "w") as f:
        json.dump({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_name":    MODEL_NAME,
            "sampling_n":    SAMPLING_N,
            "top_heads_ranked": TOP_HEADS_RANKED,
            "k_values":      K_VALUES,
            "pair_results":  all_results,
            "aggregate_by_k": agg_by_k,
        }, f, indent=2)
    print(f"\nSaved {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ks     = K_VALUES
    rates  = [agg_by_k[k] for k in ks]

    # Per-pair lines
    for pr in all_results:
        pair_rates = [next(r for r in pr["k_results"] if r["k"] == k)["corrupt_rate"]
                      for k in ks]
        ax.plot(ks, pair_rates, marker="o", alpha=0.3, linewidth=1,
                label=pr["pair_id"])

    # Aggregate
    ax.plot(ks, rates, marker="o", linewidth=2.5, color="black", label="Mean")
    ax.axhline(sum(pr["baseline_corrupt_rate"] for pr in all_results) / len(all_results),
               linestyle="--", color="gray", linewidth=1, label="Baseline corrupt rate")

    ax.set_xlabel("k (number of heads patched)")
    ax.set_ylabel("Corrupt rhyme rate")
    ax.set_title("Gemma-3-27B — Top-k head patching at newline position (i=0)\n"
                 "corrupt → clean, 5 pairs averaged")
    ax.set_xticks(ks)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_png = os.path.join(OUT_DIR, "topk_corrupt_rate.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved {out_png}")
    print("\nDone.")


if __name__ == "__main__":
    main()
