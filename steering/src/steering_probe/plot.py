"""
Plotting utilities: layer × position heatmaps of steered_rhyme_pct.
"""
import json
import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _build_matrix(
    layer_dict: dict,
    layers: List[int],
    positions: List[int],
) -> np.ndarray:
    """Build a (n_layers, n_positions) matrix of steered_rhyme_pct."""
    mat = np.full((len(layers), len(positions)), np.nan)
    for li, l in enumerate(layers):
        for pi, p in enumerate(positions):
            entry = layer_dict.get(str(l), {}).get(str(p))
            if entry is not None:
                mat[li, pi] = entry.get("steered_rhyme_pct", np.nan)
    return mat


def plot_pair_heatmap(
    layer_dict: dict,
    layers: List[int],
    positions: List[int],
    title: str,
    output_path: str,
    baseline: Optional[float] = None,
) -> None:
    mat = _build_matrix(layer_dict, layers, positions)

    fig, ax = plt.subplots(figsize=(max(8, len(positions) * 0.5 + 2), max(5, len(layers) * 0.15 + 2)))
    im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0, cmap="RdYlGn", origin="lower")

    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([str(p) for p in positions], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([str(l) for l in layers], fontsize=7)
    ax.set_xlabel("Position i  (0 = newline, negative = prompt, positive = generation)")
    ax.set_ylabel("Layer")

    full_title = title
    if baseline is not None:
        full_title += f"   [baseline rhyme%: {baseline:.1%}]"
    ax.set_title(full_title, fontsize=10)

    plt.colorbar(im, ax=ax, label="Steered Rhyme %")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_all_pairs(
    results: dict,
    scheme_names: Dict[int, str],
    output_dir: str,
) -> None:
    """
    results structure:
      {str(src): {str(tgt): {str(layer): {str(pos): {steered_rhyme_pct, baseline_rhyme_pct, n}}}}}
    """
    os.makedirs(output_dir, exist_ok=True)

    for src_str, tgt_dict in results.items():
        src = int(src_str)
        for tgt_str, layer_dict in tgt_dict.items():
            tgt = int(tgt_str)
            if not layer_dict:
                continue

            layers = sorted(int(l) for l in layer_dict)
            positions = sorted(set(
                int(p) for ld in layer_dict.values() for p in ld
            ))

            # Pull one baseline value (constant across layers/positions for same src)
            baseline = None
            for ld in layer_dict.values():
                for entry in ld.values():
                    if entry.get("baseline_rhyme_pct") is not None:
                        baseline = entry["baseline_rhyme_pct"]
                        break
                if baseline is not None:
                    break

            src_name = scheme_names.get(src, str(src))
            tgt_name = scheme_names.get(tgt, str(tgt))
            title = f"Scheme {src_name} → {tgt_name}"
            fname = f"steer_{src}_{tgt}.png"

            plot_pair_heatmap(
                layer_dict, layers, positions, title,
                os.path.join(output_dir, fname),
                baseline=baseline,
            )
    print(f"Heatmaps saved to {output_dir}")

    # --- Bar graphs Nick requested ---
    plot_per_position_layer_bars(results, scheme_names, output_dir)
    plot_summary_position_bar(results, scheme_names, output_dir)


def _collect_all_results(results: dict):
    """Yield (src, tgt, layer, pos, entry) for every data point."""
    for src_str, tgt_dict in results.items():
        for tgt_str, layer_dict in tgt_dict.items():
            for layer_str, pos_dict in layer_dict.items():
                for pos_str, entry in pos_dict.items():
                    yield int(src_str), int(tgt_str), int(layer_str), int(pos_str), entry


def plot_per_position_layer_bars(
    results: dict,
    scheme_names: Dict[int, str],
    output_dir: str,
) -> None:
    """
    For each position: bar graph with x=layer, y=mean steered_rhyme_accuracy across all pairs.
    """
    from collections import defaultdict

    # Gather: position -> layer -> list of steered_rhyme_pct
    pos_layer_vals: Dict[int, Dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for src, tgt, layer, pos, entry in _collect_all_results(results):
        val = entry.get("steered_rhyme_pct")
        if val is not None:
            pos_layer_vals[pos][layer].append(val)

    bar_dir = os.path.join(output_dir, "bar_graphs")
    os.makedirs(bar_dir, exist_ok=True)

    for pos in sorted(pos_layer_vals):
        layer_data = pos_layer_vals[pos]
        layers = sorted(layer_data.keys())
        means = [np.mean(layer_data[l]) for l in layers]

        fig, ax = plt.subplots(figsize=(max(10, len(layers) * 0.3), 5))
        ax.bar(range(len(layers)), means, color="steelblue")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([str(l) for l in layers], rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean Steered Rhyme Accuracy")
        ax.set_ylim(0, 1)
        ax.set_title(f"Position {pos}: Steered Rhyme Accuracy by Layer (avg across all pairs)")
        plt.tight_layout()
        fname = f"bar_pos_{pos}.png"
        plt.savefig(os.path.join(bar_dir, fname), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Per-position bar graphs saved to {bar_dir}")


def plot_summary_position_bar(
    results: dict,
    scheme_names: Dict[int, str],
    output_dir: str,
) -> None:
    """
    Summary bar graph: x=position, y=max accuracy across layers (averaged across pairs).
    For each position, find the best layer (highest mean across pairs), plot that value.
    """
    from collections import defaultdict

    # position -> layer -> list of steered_rhyme_pct
    pos_layer_vals: Dict[int, Dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for src, tgt, layer, pos, entry in _collect_all_results(results):
        val = entry.get("steered_rhyme_pct")
        if val is not None:
            pos_layer_vals[pos][layer].append(val)

    positions = sorted(pos_layer_vals.keys())
    max_means = []
    best_layers = []
    for pos in positions:
        layer_data = pos_layer_vals[pos]
        best_layer = max(layer_data, key=lambda l: np.mean(layer_data[l]))
        best_mean = np.mean(layer_data[best_layer])
        max_means.append(best_mean)
        best_layers.append(best_layer)

    fig, ax = plt.subplots(figsize=(max(8, len(positions) * 0.8), 5))
    bars = ax.bar(range(len(positions)), max_means, color="darkorange")
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([str(p) for p in positions])
    ax.set_xlabel("Position")
    ax.set_ylabel("Max Mean Steered Rhyme Accuracy (best layer)")
    ax.set_ylim(0, 1)
    ax.set_title("Summary: Best Layer Accuracy by Position")

    # Annotate bars with the best layer number
    for i, (bar, bl) in enumerate(zip(bars, best_layers)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"L{bl}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "summary_position_bar.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Summary bar graph saved to {out_path}")
