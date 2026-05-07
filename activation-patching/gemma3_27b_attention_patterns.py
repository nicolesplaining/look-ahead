"""
Attention pattern visualization for Gemma-3-27B.

For each prompt pair, runs a forward pass with output_attentions=True and
extracts attention weights in layers 31-41 (the "critical window" where
information moves from i=-2 to i=0).

Plots:
  1. Per-layer heatmap of attention FROM the newline token (i=0) TO all positions,
     averaged across the 5 prompt pairs and both clean/corrupt prompts.
  2. How much i=0 attends to i=-2 (rhyme word) vs other positions, per layer and head.

Usage:
    python gemma3_27b_attention_patterns.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME  = "google/gemma-3-27b-it"
LAYER_START = 27   # inclusive
LAYER_END   = 45   # inclusive
OUT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "gemma3_27b_attention_patterns")

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

OFFSETS = [-2, -1, 0, 1, 2, 3]

# ── Helpers ───────────────────────────────────────────────────────────────────

def find_positions(tokenizer, prompt):
    """Return dict of offset → abs token position relative to second newline."""
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    ids = enc["input_ids"]
    om  = enc["offset_mapping"]

    nl_chars = [i for i, ch in enumerate(prompt) if ch == "\n"]
    second_nl_char = nl_chars[1]
    second_nl_tok  = next(i for i, (s, e) in enumerate(om) if s <= second_nl_char < e)

    positions = {}
    for off in OFFSETS:
        p = second_nl_tok + off
        if 0 <= p < len(ids):
            positions[off] = p
    return positions, ids, second_nl_tok

def get_attention(model, tokenizer, prompt, layers):
    """Run forward pass and return attention weights for specified layers."""
    device = model.model.language_model.embed_tokens.weight.device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc, output_attentions=True)
    # out.attentions: tuple of (1, n_heads, seq, seq) per layer
    return [out.attentions[l].squeeze(0).cpu().float().numpy() for l in layers]

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    n_layers = model.config.text_config.num_hidden_layers
    n_heads  = model.config.text_config.num_attention_heads
    print(f"Layers: {n_layers} | Heads: {n_heads}")

    layers = list(range(LAYER_START, min(LAYER_END + 1, n_layers)))

    # Collect attention from i=0 → i=-2 for each (pair, prompt_type, layer, head)
    # Shape accumulator: [n_layers_window, n_heads]
    attn_nl_to_rhyme_clean   = np.zeros((len(layers), n_heads))  # newline → rhyme word
    attn_nl_to_rhyme_corrupt = np.zeros((len(layers), n_heads))
    count = 0

    # Also collect full attention rows from i=0 for heatmap (avg over pairs/prompts)
    # We'll store per-pair for plotting
    all_results = []

    for pair in PROMPT_PAIRS:
        pair_id = pair["pair_id"]
        print(f"\nProcessing {pair_id}...")

        for prompt_type in ["clean", "corrupt"]:
            prompt = pair[f"{prompt_type}_prompt"]
            positions, ids, second_nl_tok = find_positions(tokenizer, prompt)

            nl_pos   = positions.get(0)   # newline token (i=0)
            rhyme_pos = positions.get(-2)  # rhyme word token (i=-2)

            if nl_pos is None or rhyme_pos is None:
                print(f"  Skipping {prompt_type}: couldn't resolve positions")
                continue

            tok_labels = [tokenizer.decode([t]) for t in ids]
            print(f"  {prompt_type}: nl_pos={nl_pos} ({repr(tok_labels[nl_pos])}), "
                  f"rhyme_pos={rhyme_pos} ({repr(tok_labels[rhyme_pos])}), "
                  f"seq_len={len(ids)}")

            attn_layers = get_attention(model, tokenizer, prompt, layers)
            # attn_layers[l]: [n_heads, seq, seq]

            for l_idx, attn in enumerate(attn_layers):
                # attention FROM nl_pos TO rhyme_pos, across all heads
                nl_row = attn[:, nl_pos, :]  # [n_heads, seq_len]
                if prompt_type == "clean":
                    attn_nl_to_rhyme_clean[l_idx]   += nl_row[:, rhyme_pos]
                else:
                    attn_nl_to_rhyme_corrupt[l_idx] += nl_row[:, rhyme_pos]

            # Save full attention row from nl_pos for this pair/prompt (avg over heads)
            all_results.append({
                "pair_id":     pair_id,
                "prompt_type": prompt_type,
                "nl_pos":      nl_pos,
                "rhyme_pos":   rhyme_pos,
                "seq_len":     len(ids),
                "tok_labels":  tok_labels,
                "positions":   positions,
                # avg over heads, shape [n_layers_window, seq_len]
                "avg_attn_from_nl": np.stack([
                    attn[:, nl_pos, :].mean(axis=0) for attn in attn_layers
                ]),
            })

        count += 1

    # Average over pairs
    attn_nl_to_rhyme_clean   /= count
    attn_nl_to_rhyme_corrupt /= count

    # ── Plot 1: heatmap of attention from i=0 → i=-2, per (layer, head) ──────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, data, title in zip(
        axes,
        [attn_nl_to_rhyme_clean, attn_nl_to_rhyme_corrupt],
        ["Clean prompts", "Corrupt prompts"],
    ):
        im = ax.imshow(data, aspect="auto", cmap="Reds",
                       origin="upper", vmin=0)
        ax.set_xlabel("Head index")
        ax.set_ylabel("Layer")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([str(l) for l in layers], fontsize=7)
        ax.set_title(f"{title}\nAttention from i=0 (newline) → i=-2 (rhyme word)")
        plt.colorbar(im, ax=ax)

    fig.suptitle(f"Gemma-3-27B — Attention weight: newline → rhyme word\n"
                 f"Layers {LAYER_START}–{LAYER_END}, averaged over 5 prompt pairs",
                 fontsize=11)
    plt.tight_layout()
    out1 = os.path.join(OUT_DIR, "attn_nl_to_rhyme_heatmap.png")
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"\nSaved {out1}")

    # ── Plot 2: avg attention weight from i=0 per layer (summed over heads) ──
    fig, ax = plt.subplots(figsize=(8, 5))
    clean_per_layer   = attn_nl_to_rhyme_clean.sum(axis=1)    # sum over heads
    corrupt_per_layer = attn_nl_to_rhyme_corrupt.sum(axis=1)
    ax.plot(layers, clean_per_layer,   marker="o", label="Clean prompts",   color="steelblue")
    ax.plot(layers, corrupt_per_layer, marker="o", label="Corrupt prompts", color="crimson")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Total attention weight (summed over heads)")
    ax.set_title("Gemma-3-27B — How much does i=0 (newline) attend to i=-2 (rhyme word)?")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, "attn_nl_to_rhyme_by_layer.png")
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"Saved {out2}")

    # ── Plot 3: full attention row from i=0, avg over pairs, per layer ────────
    # Show where the newline token attends (relative positions -4 to +4)
    # Use clean prompts, avg over pairs
    clean_results = [r for r in all_results if r["prompt_type"] == "clean"]
    offset_range  = list(range(-4, 5))

    fig, axes = plt.subplots(4, 5, figsize=(16, 12), sharey=False)
    axes = axes.flatten()
    for l_idx, layer in enumerate(layers):
        if l_idx >= len(axes):
            break
        ax = axes[l_idx]
        # avg attention from nl across pairs, at this layer, for offset positions
        vals = []
        for off in offset_range:
            attn_vals = []
            for r in clean_results:
                pos = r["positions"].get(off)
                if pos is not None:
                    attn_vals.append(r["avg_attn_from_nl"][l_idx, pos])
            vals.append(np.mean(attn_vals) if attn_vals else 0.0)

        bars = ax.bar(offset_range, vals, color=[
            "crimson" if o == -2 else "steelblue" if o == 0 else "lightgray"
            for o in offset_range
        ])
        ax.set_title(f"Layer {layer}", fontsize=8)
        ax.set_xticks(offset_range)
        ax.set_xticklabels([str(o) for o in offset_range], fontsize=6)
        ax.tick_params(labelsize=6)
        ax.set_ylim(0, None)

    for i in range(len(layers), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Gemma-3-27B — Attention from i=0 (newline) to nearby positions\n"
                 "Red = i=-2 (rhyme word), Blue = i=0 itself | Clean prompts, avg over 5 pairs",
                 fontsize=10)
    plt.tight_layout()
    out3 = os.path.join(OUT_DIR, "attn_from_nl_by_position.png")
    plt.savefig(out3, dpi=150)
    plt.close()
    print(f"Saved {out3}")

    # Save summary JSON
    summary = {
        "model": MODEL_NAME,
        "layers": layers,
        "n_heads": n_heads,
        "attn_nl_to_rhyme_clean":   attn_nl_to_rhyme_clean.tolist(),
        "attn_nl_to_rhyme_corrupt": attn_nl_to_rhyme_corrupt.tolist(),
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
