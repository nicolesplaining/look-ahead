"""
Standalone panel (a) for Fig. 7: heatmap of attention weight from the newline
token (i=0) to the last word token (i=-2), per (layer, head), in Gemma-3-27B
layers 27-45. Red stars mark the top-5 heads by attention weight (the same
heads patched in panel (b)).

Output:
  icml2026/media/results-patching/topk_head_attn_panel_a-updated.png

Data source:
  activation-patching/results/gemma3_27b_attention_patterns/summary.json
  (19 layers x 32 heads, averaged across 5 prompt pairs, clean prompts)
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(
    ROOT, "results", "gemma3_27b_attention_patterns", "summary.json"
)
OUT = os.path.join(
    ROOT, "..", "icml2026", "media", "results-patching",
    "topk_head_attn_panel_a-updated.png",
)


def main():
    with open(SUMMARY) as f:
        d = json.load(f)
    layers = d["layers"]
    n_heads = d["n_heads"]

    # shape (n_layers, n_heads); we want (heads on y, layers on x) → transpose
    mat = np.array(d["attn_nl_to_rhyme_clean"]).T   # (n_heads, n_layers)

    # top-5 heads by attention weight
    flat = mat.flatten()
    top5_flat = np.argsort(flat)[-5:][::-1]
    top5 = [(int(i // mat.shape[1]), int(i % mat.shape[1])) for i in top5_flat]
    print("top-5 (head, layer-idx, layer, weight):")
    for h, lidx in top5:
        print(f"  head={h:2d}  layer={layers[lidx]:2d}  weight={mat[h, lidx]:.3f}")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(mat, aspect="auto", cmap="Blues", origin="lower",
                   vmin=0, vmax=1.0)

    star_x = [lidx for (_, lidx) in top5]
    star_y = [h    for (h, _) in top5]
    ax.scatter(star_x, star_y, marker="*", s=180,
               facecolor="#cc6677", edgecolor="black", linewidth=0.8,
               label="Top-5", zorder=5)

    ax.set_xticks(range(0, len(layers), 3))
    ax.set_xticklabels([str(layers[i]) for i in range(0, len(layers), 3)],
                       fontsize=10)
    ax.set_yticks(range(0, n_heads, 8))
    ax.set_yticklabels([str(i) for i in range(0, n_heads, 8)], fontsize=10)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Attention head", fontsize=12)

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(labelsize=9)

    ax.legend(loc="upper right", frameon=False, fontsize=10,
              handletextpad=0.2, borderaxespad=0.4)

    plt.tight_layout()
    plt.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
