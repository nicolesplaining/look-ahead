"""
Regenerate scaling-top5.png and scaling-rhyme1.png with Llama-3.1-70B added.

For each model, compute max(layer-wise) accuracy gap = i0_acc[layer] - i1_acc[layer].
Plot grouped bars by family (Gemma-3, Qwen3, Llama-3).
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "..", "results")
OUT_DIR = os.path.join(ROOT, "..", "..", "icml2026", "media", "results-poem")

# Map of (family, label, results_subdir). One entry per model size.
MODELS = [
    # Gemma-3
    ("Gemma-3", "1B",  "Gemma-3-1B"),
    ("Gemma-3", "4B",  "Gemma-3-4B"),
    ("Gemma-3", "12B", "Gemma-3-12B"),
    ("Gemma-3", "27B", "Gemma-3-27B"),
    # Qwen3
    ("Qwen3", "0.6B", "Qwen3-0.6B"),
    ("Qwen3", "1.7B", "Qwen3-1.7B"),
    ("Qwen3", "4B",   "Qwen3-4B"),
    ("Qwen3", "8B",   "Qwen3-8B"),
    ("Qwen3", "14B",  "Qwen3-14B"),
    ("Qwen3", "32B",  "qwen3-32B"),
    # Llama-3
    ("Llama-3.1/3.2", "1B",  "Llama-3.2-1B"),
    ("Llama-3.1/3.2", "3B",  "Llama-3.2-3B"),
    ("Llama-3.1/3.2", "8B",  "Llama-3.1-8B"),
    ("Llama-3.1/3.2", "70B", "Llama-3.1-70B-Instruct"),
]

FAMILY_COLORS = {
    "Gemma-3": "#4477aa",
    "Qwen3":   "#cc6677",
    "Llama-3.1/3.2": "#117733",
}


def load_layer_accs(model_dir, pos_dir, key):
    path = os.path.join(RES, model_dir, pos_dir, "experiment_results.json")
    with open(path) as f:
        d = json.load(f)
    res = d["results"]
    # entries are layerN_i<P>; pull out the right ones in layer order
    layers = []
    for k, v in res.items():
        layers.append((v["layer"], v[key]))
    layers.sort()
    return [a for _, a in layers]


def max_gap(model_dir, key):
    """Max over layers of (i=0 acc - i=1 acc)."""
    a0 = load_layer_accs(model_dir, "i0", key)
    a1 = load_layer_accs(model_dir, "i1", key)
    n = min(len(a0), len(a1))
    return max(a0[i] - a1[i] for i in range(n))


def plot_scaling(metric_key, out_filename, ylabel):
    # Group by family
    fam_data = {}
    for fam, lab, mdir in MODELS:
        try:
            gap = max_gap(mdir, metric_key)
        except Exception as e:
            print(f"  SKIP {fam} {lab} ({mdir}): {e}")
            continue
        fam_data.setdefault(fam, []).append((lab, gap))

    n_fams = len(fam_data)
    fig, axes = plt.subplots(1, n_fams, figsize=(4 * n_fams, 3.2), sharey=True)
    if n_fams == 1:
        axes = [axes]

    for ax, (fam, items) in zip(axes, fam_data.items()):
        labels = [lab for lab, _ in items]
        vals   = [g   for _, g in items]
        x = np.arange(len(items))
        ax.bar(x, vals, color=FAMILY_COLORS[fam], width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(fam, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, out_filename)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
    for fam, items in fam_data.items():
        for lab, gap in items:
            print(f"  {fam:<15} {lab:>5}  gap={gap:.3f}")


plot_scaling("val_top5_accuracy", "scaling-top5.png",  "Accuracy difference")
plot_scaling("top5_rhyme_accuracy", "scaling-rhyme1.png", "Accuracy difference")
