"""
Regenerate the three main per-layer figures with consistent taller aspect ratio:
  - patching_qwen3_32b.png   (Qwen3-32B,   i=-1 + i=0)
  - patching_gemma3_27b.png  (Gemma-3-27B, i=-2 + i=0)
  - patching_llama_70b.png   (Llama-3.1-70B, i=-1 + i=0)

All saved at figsize=(8, 4) to match the Llama-70B figure.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "..", "icml2026", "media", "results-patching")

# (out_filename, agg_path, last_word_pos_id, last_word_label, fmt)
JOBS = [
    ("patching_qwen3_32b.png",
     f"{RES}/QWEN3_AGGREGATE/qwen3_32b_aggregate_N20/aggregate.json",
     "i_minus1", "i=-1", "canonical"),
    ("patching_gemma3_27b.png",
     f"{RES}/GEMMA3_AGGREGATE/gemma3_27b_aggregate_N20/aggregate.json",
     "i_minus2", "i=-2", "canonical"),
    ("patching_llama_70b.png",
     f"{RES}/llama-3.1-70b-per-layer-per-position/aggregate.json",
     "i_minus1", "i=-1", "minimal"),
]


def load_rates(path, lw_pos, fmt):
    with open(path) as f:
        d = json.load(f)
    agg = d["aggregate"]
    if fmt == "canonical":
        m1 = [r["mean_corrupt_rhyme_rate"] for r in agg[lw_pos]]
        i0 = [r["mean_corrupt_rhyme_rate"] for r in agg["i_0"]]
    else:
        m1 = [r["corrupt_rhyme_rate"] for r in agg[lw_pos]]
        i0 = [r["corrupt_rhyme_rate"] for r in agg["i_0"]]
    return m1, i0


for out_name, path, lw_pos, lw_label, fmt in JOBS:
    m1, i0 = load_rates(path, lw_pos, fmt)
    n = len(m1)
    x = np.arange(n)
    w = 0.4

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, m1, w, label=lw_label, color="#6699cc")
    ax.bar(x + w/2, i0, w, label="i=0",     color="#cc6677")
    ax.set_xlabel("Layer", fontsize=18)
    ax.set_ylabel("Corrupt rhyme rate", fontsize=18)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-1, n)
    step = max(1, n // 16)
    ax.set_xticks(np.arange(0, n, step))
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="upper right", frameon=False, fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out_path = os.path.join(OUT, out_name)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}  (peak {lw_label}: {max(m1):.3f}, peak i=0: {max(i0):.3f})")
