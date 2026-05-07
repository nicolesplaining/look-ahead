"""
Generate the per-layer corrupt-rhyme-rate figure for Llama-3.1-70B.

Produces patching_llama_70b.png, matching the style of patching_llama_8b.png:
side-by-side bars (i=-1 blue, i=0 red) for each of the 80 layers, mean across
the 5 prompt pairs.

Reads from results/llama-3.1-70b-per-layer-per-position/aggregate.json.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
AGG_PATH = os.path.join(
    ROOT, "results", "llama-3.1-70b-per-layer-per-position", "aggregate.json"
)
OUT_PATH = os.path.join(
    ROOT, "..", "icml2026", "media", "results-patching", "patching_llama_70b.png"
)

with open(AGG_PATH) as f:
    agg = json.load(f)

n_layers = agg["n_layers"]
i_minus1 = [r["corrupt_rhyme_rate"] for r in agg["aggregate"]["i_minus1"]]
i_0      = [r["corrupt_rhyme_rate"] for r in agg["aggregate"]["i_0"]]

x = np.arange(n_layers)
w = 0.4

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x - w/2, i_minus1, w, label="i=-1", color="#6699cc")
ax.bar(x + w/2, i_0,      w, label="i=0",  color="#cc6677")

ax.set_xlabel("Layer")
ax.set_ylabel("Corrupt rhyme rate")
ax.set_ylim(0, 1.05)
ax.set_xlim(-1, n_layers)
ax.set_xticks(np.arange(0, n_layers, 4))
ax.legend(loc="upper right", frameon=False)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
print(f"Saved {OUT_PATH}")

# Also report key stats
peak_minus1 = max(i_minus1)
peak_0      = max(i_0)
print(f"Llama-3.1-70B peak corrupt_rhyme_rate i=-1: {peak_minus1:.3f} (layer {i_minus1.index(peak_minus1)})")
print(f"Llama-3.1-70B peak corrupt_rhyme_rate i=0 : {peak_0:.3f} (layer {i_0.index(peak_0)})")
