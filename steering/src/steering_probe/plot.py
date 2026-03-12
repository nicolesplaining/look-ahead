"""
Plotting utilities: layer bar charts of corruption rate by position.
"""
import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# POSITIONS = [-2, 0]
# COLORS = {-2: "#4C72B0", 0: "#DD8452"}
# LABELS = {-2: "position -2 (last word token)", 0: "position 0 (newline token)"}

POSITIONS = [-1, 0]
COLORS = {-1: "#4C72B0", 0: "#DD8452"}
LABELS = {-1: "position -1 (last word token)", 0: "position 0 (newline token)"}


def plot_pair_bargraph(
    layer_dict: dict,
    layers: List[int],
    title: str,
    output_path: str,
    baseline: Optional[float] = None,
) -> None:
    """
    Bar chart: x = layer, y = steered_rhyme_pct,
    two bars per layer for positions -1 and 0.
    """
    n_layers = len(layers)
    x = np.arange(n_layers)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, n_layers * 0.4 + 2), 5))

    for i, pos in enumerate(POSITIONS):
        values = []
        for layer in layers:
            entry = layer_dict.get(str(layer), {}).get(str(pos))
            if entry is not None:
                values.append(entry.get("steered_rhyme_pct", np.nan))
            else:
                values.append(np.nan)

        offset = (i - 0.5) * width
        ax.bar(x + offset, values, width, label=LABELS[pos], color=COLORS[pos], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Proportion of Steered Rhymes")
    ax.set_ylim(0, 1.05)
    # ax.set_title(title, fontsize=11)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_all_pairs(
    results: dict,
    scheme_names: Dict[int, str],
    output_dir: str,
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """
    Produce a single aggregated plot averaging steered_rhyme_pct across all
    (src, tgt) pairs, one bar group per layer, two bars per group for pos -1 / 0.

    results structure:
      {str(src): {str(tgt): {str(layer): {str(pos): {steered_rhyme_pct, baseline_rhyme_pct, n}}}}}
    """
    from collections import defaultdict
    os.makedirs(output_dir, exist_ok=True)

    agg: Dict[int, Dict[int, list]] = defaultdict(lambda: defaultdict(list))

    for src_str, tgt_dict in results.items():
        for tgt_str, layer_dict in tgt_dict.items():
            if not layer_dict:
                continue
            for layer_str, pos_dict in layer_dict.items():
                layer = int(layer_str)
                for pos_str, entry in pos_dict.items():
                    if not isinstance(entry, dict):
                        continue
                    pos = int(pos_str)
                    if pos not in POSITIONS:
                        continue
                    v = entry.get("steered_rhyme_pct")
                    if v is not None:
                        agg[layer][pos].append(v)

    layers = sorted(agg.keys())
    n_pairs = max(
        (len(vals) for pos_dict in agg.values() for vals in pos_dict.values()),
        default=1,
    )

    n_layers = len(layers)
    x = np.arange(n_layers)
    width = 0.35

    default_figsize = (max(10, n_layers * 0.4 + 2), 5)
    fig, ax = plt.subplots(figsize=figsize or default_figsize)

    for i, pos in enumerate(POSITIONS):
        means = [float(np.mean(agg[l][pos])) if agg[l][pos] else np.nan for l in layers]
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width, label=LABELS[pos], color=COLORS[pos], alpha=0.85)

    tick_labels = [str(l) if i % 5 == 0 else "" for i, l in enumerate(layers)]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel(xlabel or "Layer")
    ax.set_ylabel("Fraction of \n Steered Rhyme", fontsize=8)
    ax.set_ylim(0, 1.05)
    # ax.set_title(title or f"Steering effectiveness aggregated over {n_pairs} pairs", fontsize=11)
    ax.legend()

    plt.tight_layout()
    out = os.path.join(output_dir, "steer_aggregated.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Aggregated plot saved → {out}")
