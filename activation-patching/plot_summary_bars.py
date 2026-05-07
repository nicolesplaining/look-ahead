"""
Regenerate patching_summary_bars.png with Llama-3.1-70B added.

Shows peak corrupt rhyme rate (max across all layers) at i=-1 (or i=-2 for
Gemma) and i=0 for each model size, grouped by family.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
OUT_PATH = os.path.join(
    ROOT, "..", "icml2026", "media", "results-patching", "patching_summary_bars.png"
)

# (family, label, agg_path, last_word_pos_id, json_format)
MODELS = [
    ("Qwen3", "0.6B",  f"{RES}/AGGREGATE/Qwen_Qwen3-0.6B_aggregate_N20/aggregate.json", "i_minus1", "canonical"),
    ("Qwen3", "1.7B",  f"{RES}/AGGREGATE/Qwen_Qwen3-1.7B_aggregate_N20/aggregate.json", "i_minus1", "canonical"),
    ("Qwen3", "4B",    f"{RES}/AGGREGATE/Qwen_Qwen3-4B_aggregate_N20/aggregate.json",   "i_minus1", "canonical"),
    ("Qwen3", "8B",    f"{RES}/AGGREGATE/Qwen_Qwen3-8B_aggregate_N20/aggregate.json",   "i_minus1", "canonical"),
    ("Qwen3", "14B",   f"{RES}/AGGREGATE/Qwen_Qwen3-14B_aggregate_N20/aggregate.json",  "i_minus1", "canonical"),
    ("Qwen3", "32B",   f"{RES}/QWEN3_AGGREGATE/qwen3_32b_aggregate_N20/aggregate.json", "i_minus1", "canonical"),
    ("Gemma-3", "1B",  f"{RES}/AGGREGATE/google_gemma-3-1b-it_aggregate_N20/aggregate.json",  "i_minus2", "canonical"),
    ("Gemma-3", "4B",  f"{RES}/AGGREGATE/google_gemma-3-4b-it_aggregate_N20/aggregate.json",  "i_minus2", "canonical"),
    ("Gemma-3", "12B", f"{RES}/AGGREGATE/google_gemma-3-12b-it_aggregate_N20/aggregate.json", "i_minus2", "canonical"),
    ("Gemma-3", "27B", f"{RES}/GEMMA3_AGGREGATE/gemma3_27b_aggregate_N20/aggregate.json",     "i_minus2", "canonical"),
    ("Llama-3", "1B",  f"{RES}/AGGREGATE/meta-llama_Llama-3.2-1B-Instruct_aggregate_N20/aggregate.json", "i_minus1", "canonical"),
    ("Llama-3", "3B",  f"{RES}/AGGREGATE/meta-llama_Llama-3.2-3B-Instruct_aggregate_N20/aggregate.json", "i_minus1", "canonical"),
    ("Llama-3", "8B",  f"{RES}/AGGREGATE/meta-llama_Llama-3.1-8B-Instruct_aggregate_N20/aggregate.json", "i_minus1", "canonical"),
    ("Llama-3", "70B", f"{RES}/llama-3.1-70b-per-layer-per-position/aggregate.json",                     "i_minus1", "minimal"),
]


def peaks(path, last_word_pos, fmt):
    with open(path) as f:
        d = json.load(f)
    agg = d["aggregate"]
    if fmt == "canonical":
        m1 = max(r["mean_corrupt_rhyme_rate"] for r in agg[last_word_pos])
        i0 = max(r["mean_corrupt_rhyme_rate"] for r in agg["i_0"])
    else:  # minimal
        m1 = max(r["corrupt_rhyme_rate"] for r in agg[last_word_pos])
        i0 = max(r["corrupt_rhyme_rate"] for r in agg["i_0"])
    return m1, i0


peak_lw = []
peak_nl = []
labels  = []
families = []
for fam, sz, path, lw_pos, fmt in MODELS:
    m1, i0 = peaks(path, lw_pos, fmt)
    peak_lw.append(m1)
    peak_nl.append(i0)
    labels.append(sz)
    families.append(fam)
    print(f"{fam:>8} {sz:>5}  i=lw: {m1:.3f}   i=0: {i0:.3f}")

# Plot grouped bars
fig, ax = plt.subplots(figsize=(11, 3.5))
x = np.arange(len(MODELS))
w = 0.4
ax.bar(x - w/2, peak_lw, w, label="last word", color="#6699cc")
ax.bar(x + w/2, peak_nl, w, label="i=0 (newline)", color="#cc6677")

# X tick labels include family above first model in each group
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)

# Family group separators + labels
boundaries = []
prev = None
for i, fam in enumerate(families):
    if fam != prev:
        boundaries.append(i)
    prev = fam
boundaries.append(len(families))

for j in range(len(boundaries) - 1):
    s = boundaries[j]
    e = boundaries[j + 1]
    fam = families[s]
    mid = (s + e - 1) / 2
    ax.text(mid, 1.07, fam, ha="center", va="bottom", fontsize=10,
            transform=ax.get_xaxis_transform())
    if j > 0:
        ax.axvline(s - 0.5, color="gray", linestyle=":", alpha=0.5)

ax.set_ylim(0, 1.05)
ax.set_ylabel("Peak corrupt rhyme rate")
ax.legend(loc="upper right", frameon=False, fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
print(f"\nSaved {OUT_PATH}")
