"""
Top-k MLP layer patching at the newline position (i=0) for Gemma-3-27B.

Step 1: For each MLP layer, compute the L2 norm of the difference between
        corrupt and clean MLP outputs at i=0 (newline token).
        This identifies which layers are most sensitive to the rhyme word.

Step 2: Patch the top-k MLP layers simultaneously (corrupt -> clean)
        at i=0 and measure corrupt rhyme rate.

Tries k = 1, 2, 3, 5, 10 layers.

Usage:
    python gemma3_27b_topk_mlp_patching.py
"""

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import pronouncing
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME     = "google/gemma-3-27b-it"
SAMPLING_N     = 20
SAMPLING_TEMP  = 0.7
MAX_NEW_TOKENS = 20
BATCH_SIZE     = 20

K_VALUES = [1, 2, 3, 5, 10]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "gemma3_27b_topk_mlp_patching")

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

# ── MLP output caching ────────────────────────────────────────────────────────

def cache_mlp_outputs(model, tokenizer, prompt, nl_pos, device):
    """Cache MLP output at nl_pos for every layer."""
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    n_layers = model.config.text_config.num_hidden_layers
    cached = [None] * n_layers
    handles = []

    def make_capture(idx):
        def hook(module, args, output):
            if output.shape[1] > nl_pos:
                cached[idx] = output[0, nl_pos, :].detach().clone()
        return hook

    for idx in range(n_layers):
        h = model.model.language_model.layers[idx].mlp.register_forward_hook(
            make_capture(idx)
        )
        handles.append(h)

    try:
        with torch.no_grad():
            model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    return cached

# ── MLP patching ──────────────────────────────────────────────────────────────

@contextmanager
def patch_mlp_layers(model, corrupt_cache, top_layers, nl_pos):
    """Replace MLP output at nl_pos for specified layers with corrupt values."""
    handles = []
    for layer_idx in top_layers:
        patch_vec = corrupt_cache[layer_idx]

        def make_hook(patch_vec, nl_pos):
            def hook(module, args, output):
                if output.shape[1] <= nl_pos:
                    return output
                out = output.clone()
                out[:, nl_pos, :] = patch_vec.to(out.device, dtype=out.dtype)
                return out
            return hook

        h = model.model.language_model.layers[layer_idx].mlp.register_forward_hook(
            make_hook(patch_vec, nl_pos)
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
    n_layers = model.config.text_config.num_hidden_layers
    device   = model.model.language_model.embed_tokens.weight.device
    print(f"Loaded. Layers={n_layers}")

    # ── Step 1: rank MLP layers by activation difference at i=0 ──────────────
    print("\n-- Step 1: Computing MLP activation differences at i=0 --")

    # Accumulate L2 norms across pairs
    l2_norms = np.zeros(n_layers)
    for pair in PROMPT_PAIRS:
        nl_pos_clean   = find_nl_pos(tokenizer, pair["clean_prompt"])
        nl_pos_corrupt = find_nl_pos(tokenizer, pair["corrupt_prompt"])
        # Use clean nl_pos as reference (should match corrupt)
        nl_pos = nl_pos_clean

        clean_cache   = cache_mlp_outputs(model, tokenizer, pair["clean_prompt"],   nl_pos, device)
        corrupt_cache = cache_mlp_outputs(model, tokenizer, pair["corrupt_prompt"], nl_pos_corrupt, device)

        for idx in range(n_layers):
            if clean_cache[idx] is not None and corrupt_cache[idx] is not None:
                diff = (corrupt_cache[idx] - clean_cache[idx].to(corrupt_cache[idx].device)).float()
                l2_norms[idx] += diff.norm().item()

        print(f"  {pair['pair_id']}: done")

    l2_norms /= len(PROMPT_PAIRS)

    # Rank layers
    ranked_layers = [int(x) for x in np.argsort(l2_norms)[::-1]]
    print("\nTop 15 MLP layers by mean L2 difference at i=0:")
    for rank, layer_idx in enumerate(ranked_layers[:15]):
        print(f"  #{rank+1:2d}  Layer {layer_idx:2d}: L2={l2_norms[layer_idx]:.4f}")

    # Save ranking
    with open(os.path.join(OUT_DIR, "mlp_layer_ranking.json"), "w") as f:
        json.dump({
            "ranked_layers": ranked_layers,
            "l2_norms": l2_norms.tolist(),
        }, f, indent=2)

    # Plot L2 norms per layer
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(n_layers), l2_norms, color="steelblue", alpha=0.7)
    for k in [1, 2, 3, 5, 10]:
        if k <= len(ranked_layers):
            ax.axvline(ranked_layers[k-1], color="red", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean L2 norm of MLP output difference")
    ax.set_title("Gemma-3-27B — MLP activation difference at i=0 (newline)\n"
                 "corrupt vs clean, averaged over 5 pairs")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mlp_l2_diff_by_layer.png"), dpi=150)
    plt.close()
    print(f"Saved mlp_l2_diff_by_layer.png")

    # ── Step 2: patch top-k MLP layers ───────────────────────────────────────
    print("\n-- Step 2: Patching top-k MLP layers --")

    all_results = []

    for pair in PROMPT_PAIRS:
        pair_id        = pair["pair_id"]
        clean_prompt   = pair["clean_prompt"]
        corrupt_prompt = pair["corrupt_prompt"]
        clean_rw       = pair["clean_rhyme_word"]
        corrupt_rw     = pair["corrupt_rhyme_word"]

        print(f"\n{'='*60}")
        print(f"Pair: {pair_id}  ({clean_rw!r} vs {corrupt_rw!r})")

        nl_pos_clean   = find_nl_pos(tokenizer, clean_prompt)
        nl_pos_corrupt = find_nl_pos(tokenizer, corrupt_prompt)

        # Baseline
        baseline = sample_completions(model, tokenizer, clean_prompt,
                                      SAMPLING_N, SAMPLING_TEMP, device)
        bl_clean_rate   = rhyme_rate(baseline, clean_prompt, clean_rw)
        bl_corrupt_rate = rhyme_rate(baseline, clean_prompt, corrupt_rw)
        print(f"  Baseline: clean={bl_clean_rate:.3f}  corrupt={bl_corrupt_rate:.3f}")

        # Cache corrupt MLP outputs
        corrupt_cache = cache_mlp_outputs(model, tokenizer, corrupt_prompt,
                                          nl_pos_corrupt, device)

        pair_results = {
            "pair_id": pair_id,
            "baseline_clean_rate":   bl_clean_rate,
            "baseline_corrupt_rate": bl_corrupt_rate,
            "k_results": [],
        }

        for k in K_VALUES:
            top_k_layers = ranked_layers[:k]
            label = ", ".join(f"L{l}" for l in top_k_layers)
            print(f"\n  k={k}: {label}")

            with patch_mlp_layers(model, corrupt_cache, top_k_layers, nl_pos_clean):
                completions = sample_completions(model, tokenizer, clean_prompt,
                                                 SAMPLING_N, SAMPLING_TEMP, device)

            cr  = rhyme_rate(completions, clean_prompt, corrupt_rw)
            clr = rhyme_rate(completions, clean_prompt, clean_rw)
            delta = cr - bl_corrupt_rate
            print(f"    corrupt_rate={cr:.3f}  clean_rate={clr:.3f}  delta={delta:+.3f}")

            pair_results["k_results"].append({
                "k":            k,
                "layers":       top_k_layers,
                "corrupt_rate": cr,
                "clean_rate":   clr,
                "delta":        delta,
            })

        all_results.append(pair_results)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("AGGREGATE SUMMARY")
    agg_by_k = {}
    for k in K_VALUES:
        rates = [next(r for r in pr["k_results"] if r["k"] == k)["corrupt_rate"]
                 for pr in all_results]
        agg_by_k[k] = sum(rates) / len(rates)
        print(f"  k={k:2d}: mean_corrupt_rate={agg_by_k[k]:.3f}")

    # Save JSON
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump({
            "timestamp_utc":  datetime.now(timezone.utc).isoformat(),
            "model_name":     MODEL_NAME,
            "sampling_n":     SAMPLING_N,
            "ranked_layers":  ranked_layers,
            "k_values":       K_VALUES,
            "pair_results":   all_results,
            "aggregate_by_k": agg_by_k,
        }, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for pr in all_results:
        pair_rates = [next(r for r in pr["k_results"] if r["k"] == k)["corrupt_rate"]
                      for k in K_VALUES]
        ax.plot(K_VALUES, pair_rates, marker="o", alpha=0.3, linewidth=1,
                label=pr["pair_id"])
    ax.plot(K_VALUES, [agg_by_k[k] for k in K_VALUES],
            marker="o", linewidth=2.5, color="black", label="Mean")
    ax.axhline(sum(pr["baseline_corrupt_rate"] for pr in all_results) / len(all_results),
               linestyle="--", color="gray", linewidth=1, label="Baseline")
    ax.set_xlabel("k (number of MLP layers patched)")
    ax.set_ylabel("Corrupt rhyme rate")
    ax.set_title("Gemma-3-27B — Top-k MLP patching at newline position (i=0)\n"
                 "corrupt → clean, 5 pairs averaged")
    ax.set_xticks(K_VALUES)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "topk_corrupt_rate.png"), dpi=150)
    plt.close()
    print(f"\nSaved results to {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
