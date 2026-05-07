"""
Regenerate appendix_patching_Llama3.png with all four Llama-3 sizes:
1B, 3B, 8B, 70B. Each panel is a per-layer bar chart at i=-1 and i=0.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(
    ROOT, "..", "icml2026", "media", "results-patching", "appendix_patching_Llama3.png"
)

# (label, aggregate_json_path, position_keys_for_minus1_and_0)
MODELS = [
    ("Llama-3.2-1B-Instruct",
     os.path.join(ROOT, "results/AGGREGATE/meta-llama_Llama-3.2-1B-Instruct_aggregate_N20/aggregate.json"),
     "canonical"),
    ("Llama-3.2-3B-Instruct",
     os.path.join(ROOT, "results/AGGREGATE/meta-llama_Llama-3.2-3B-Instruct_aggregate_N20/aggregate.json"),
     "canonical"),
    ("Llama-3.1-8B-Instruct",
     os.path.join(ROOT, "results/AGGREGATE/meta-llama_Llama-3.1-8B-Instruct_aggregate_N20/aggregate.json"),
     "canonical"),
    ("Llama-3.1-70B-Instruct",
     os.path.join(ROOT, "results/llama-3.1-70b-per-layer-per-position/aggregate.json"),
     "minimal"),
]


def load_rates(path, fmt):
    """Return (i_minus1, i_0) arrays of corrupt_rhyme_rate per layer."""
    with open(path) as f:
        d = json.load(f)
    agg = d["aggregate"]
    if fmt == "canonical":
        m1 = [r["mean_corrupt_rhyme_rate"] for r in agg["i_minus1"]]
        i0 = [r["mean_corrupt_rhyme_rate"] for r in agg["i_0"]]
    else:  # minimal (our 70B format)
        m1 = [r["corrupt_rhyme_rate"] for r in agg["i_minus1"]]
        i0 = [r["corrupt_rhyme_rate"] for r in agg["i_0"]]
    return m1, i0


fig, axes = plt.subplots(len(MODELS), 1, figsize=(10, 2.0 * len(MODELS)),
                          sharey=True)

for ax, (label, path, fmt) in zip(axes, MODELS):
    m1, i0 = load_rates(path, fmt)
    n = len(m1)
    x = np.arange(n)
    w = 0.4
    ax.bar(x - w/2, m1, w, label="i=-1", color="#6699cc")
    ax.bar(x + w/2, i0, w, label="i=0",  color="#cc6677")
    ax.set_title(label, fontsize=10, loc="left")
    ax.set_ylabel("Corrupt rhyme rate", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-1, n)
    step = max(1, n // 20)
    ax.set_xticks(np.arange(0, n, step))
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[-1].set_xlabel("Layer", fontsize=10)
axes[0].legend(loc="upper right", frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
print(f"Saved {OUT_PATH}")
