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
            positions = sorted(
                int(p) for ld in layer_dict.values() for p in ld
            )

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
    print(f"Plots saved to {output_dir}")
