"""
Plot aggregate activation-patching results.

Combines:
  - 4 new prompt pairs from aggregate experiment (N=20 each)
  - fear/fright pair from QWEN3_PER_LAYER / GEMMA3_PER_LAYER (first 20 of N=100)

Total: 5 pairs x 20 samples = 100 samples per layer per position.

For each model, produces:
  - One plot with 6 subplots (one per position), showing mean corrupt_rhyme_rate per layer.
  - One overlay plot showing all 6 positions on a single axes.
"""

import json
import os
import pronouncing
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N_FEAR = 20  # how many samples to take from the fear/fright runs

# -- Position mapping: pos_id -> fear/fright experiment subdirectory name -------

FEAR_DIR_MAP = {
    "i_minus2": "exp_fear_minus2",
    "i_minus1": "exp_fear_minus1",
    "i_0":      "exp_fear_newline",
    "i_plus1":  "exp_fear_and",
    "i_plus2":  "exp_fear_hoped",
    "i_plus3":  "exp_fear_that",
}

POSITIONS = ["i_minus2", "i_minus1", "i_0", "i_plus1", "i_plus2", "i_plus3"]
POS_LABELS = ["i=-2", "i=-1", "i=0\n(newline)", "i=+1", "i=+2", "i=+3"]

MODELS = [
    {
        "name":         "Gemma-3-27B-IT",
        "agg_dir":      "GEMMA3_AGGREGATE/gemma3_27b_aggregate_N20",
        "fear_prefix":  "gemma3_27b",
        "fear_subdir":  "GEMMA3_PER_LAYER",
        "out_prefix":   "gemma3_27b",
    },
    {
        "name":         "Qwen3-32B",
        "agg_dir":      "QWEN3_AGGREGATE/qwen3_32b_aggregate_N20",
        "fear_prefix":  "qwen3_32b",
        "fear_subdir":  "QWEN3_PER_LAYER",
        "out_prefix":   "qwen3_32b",
    },
]

# -- Rhyme helpers (to recompute rates from first N completions) -----------------

def _rhyme_score(w1, w2):
    p1 = pronouncing.phones_for_word(w1.lower().strip())
    p2 = pronouncing.phones_for_word(w2.lower().strip())
    if not p1 or not p2:
        return None
    rp1 = pronouncing.rhyming_part(p1[0])
    rp2 = pronouncing.rhyming_part(p2[0])
    return (rp1 == rp2) if (rp1 and rp2) else None

def last_word(text):
    for w in reversed(text.split()):
        cleaned = w.strip(".,!?\"'—;: ")
        if cleaned.isalpha():
            return cleaned.lower()
    return ""

def word_before_nth_newline(text, n):
    if n <= 0:
        return ""
    nls = [i for i, ch in enumerate(text) if ch == "\n"]
    if len(nls) < n:
        return ""
    end   = nls[n - 1]
    start = nls[n - 2] + 1 if n >= 2 else 0
    return last_word(text[start:end])

def extract_rhyme_word(full_text, prompt):
    idx = prompt.count("\n") + 1
    w = word_before_nth_newline(full_text, idx)
    if w:
        return w
    if full_text.startswith(prompt):
        return last_word(full_text[len(prompt):])
    return last_word(full_text)

def rhyme_rate(completions, prompt, rhyme_word):
    hits = sum(
        1 for c in completions
        if _rhyme_score(extract_rhyme_word(c, prompt), rhyme_word) is True
    )
    return hits / len(completions) if completions else 0.0

# -- Data loading ----------------------------------------------------------------

def load_pair_rates(agg_dir, pair_id, pos_id):
    """Load per-layer clean and corrupt rhyme rates for one pair+position."""
    path = os.path.join(agg_dir, pair_id, pos_id, "generations.json")
    with open(path) as f:
        data = json.load(f)
    clean_rates   = [r["clean_rhyme_rate"]   for r in data["results"]]
    corrupt_rates = [r["corrupt_rhyme_rate"]  for r in data["results"]]
    return clean_rates, corrupt_rates


def load_fear_both_rates(fear_dir, pos_id, n):
    """Load first-n per-layer clean and corrupt rhyme rates for fear/fright."""
    exp_id = FEAR_DIR_MAP[pos_id]
    candidates = [d for d in os.listdir(fear_dir) if exp_id in d]
    run_dir = os.path.join(fear_dir, sorted(candidates)[0])
    with open(os.path.join(run_dir, "generations.json")) as f:
        data = json.load(f)
    clean_prompt       = data["clean_prompt"]
    clean_rhyme_word   = data["clean_rhyme_word"]
    corrupt_rhyme_word = data["corrupt_rhyme_word"]
    clean_rates, corrupt_rates = [], []
    for layer_data in data["results"]:
        comps = layer_data["completions"][:n]
        clean_rates.append(rhyme_rate(comps, clean_prompt, clean_rhyme_word))
        corrupt_rates.append(rhyme_rate(comps, clean_prompt, corrupt_rhyme_word))
    return clean_rates, corrupt_rates


def load_fear_rates(fear_dir, pos_id, n):
    """
    Load the fear/fright generations.json for a given position,
    take first n completions per layer, return list of corrupt_rhyme_rate per layer.
    """
    exp_id = FEAR_DIR_MAP[pos_id]
    # find the directory matching the prefix
    candidates = [d for d in os.listdir(fear_dir) if exp_id in d]
    if not candidates:
        raise FileNotFoundError(f"No directory matching '{exp_id}' in {fear_dir}")
    run_dir = os.path.join(fear_dir, sorted(candidates)[0])
    json_path = os.path.join(run_dir, "generations.json")
    with open(json_path) as f:
        data = json.load(f)

    corrupt_rhyme_word = data["corrupt_rhyme_word"]
    clean_prompt       = data["clean_prompt"]
    rates = []
    for layer_data in data["results"]:
        completions = layer_data["completions"][:n]
        rate = rhyme_rate(completions, clean_prompt, corrupt_rhyme_word)
        rates.append(rate)
    return rates


def load_aggregate_rates(agg_json_path, pos_id):
    """
    Load mean_corrupt_rhyme_rate per layer for a position from aggregate.json.
    """
    with open(agg_json_path) as f:
        data = json.load(f)
    return [r["mean_corrupt_rhyme_rate"] for r in data["aggregate"][pos_id]]


def load_aggregate_baseline(agg_dir, pos_id):
    """
    Load the mean baseline corrupt_rhyme_rate across all 4 pairs for a position.
    """
    rates = []
    for pair_id in ["doom_dread", "bliss_joy", "dark_night", "grief_pain"]:
        meta_path = os.path.join(agg_dir, pair_id, "pair_meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        rates.append(meta["baseline_corrupt_rate"])
    return sum(rates) / len(rates)


def load_fear_baseline(fear_dir, pos_id, n):
    """Baseline corrupt_rhyme_rate for fear/fright (unpatched clean run, first n)."""
    exp_id = FEAR_DIR_MAP[pos_id]
    candidates = [d for d in os.listdir(fear_dir) if exp_id in d]
    run_dir = os.path.join(fear_dir, sorted(candidates)[0])
    with open(os.path.join(run_dir, "generations.json")) as f:
        data = json.load(f)
    completions = data["baseline"]["completions"][:n]
    return rhyme_rate(completions, data["clean_prompt"], data["corrupt_rhyme_word"])


# -- Combined aggregate ----------------------------------------------------------

def compute_combined(agg_rates_4pairs, fear_rates):
    """Average across 5 pairs: 4 from aggregate (already averaged) + 1 fear."""
    # agg_rates_4pairs is already mean over 4 pairs; combine with fear
    return [(a * 4 + f) / 5 for a, f in zip(agg_rates_4pairs, fear_rates)]


# -- Plotting --------------------------------------------------------------------

def plot_histogram(layers, clean_rates, corrupt_rates,
                   baseline_clean, baseline_corrupt,
                   pos_label, model_name, out_path):
    """3-panel histogram: clean rhyme / corrupt rhyme / no rhyme per layer."""
    no_rhyme_rates   = [max(0.0, 1.0 - c - r) for c, r in zip(clean_rates, corrupt_rates)]
    baseline_norhyme = max(0.0, 1.0 - baseline_clean - baseline_corrupt)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for ax, rates, baseline, color, label in [
        (ax1, clean_rates,   baseline_clean,   "steelblue",  "clean rhyme rate"),
        (ax2, corrupt_rates, baseline_corrupt, "darkorange", "corrupt rhyme rate"),
        (ax3, no_rhyme_rates, baseline_norhyme, "slategray",  "no rhyme rate"),
    ]:
        ax.bar(layers, rates, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
        ax.axhline(baseline, color="red" if ax is ax1 else ("orange" if ax is ax2 else "black"),
                   linestyle="--", linewidth=1.5, label=f"baseline ({baseline:.3f})")
        ax.set_ylabel(label, fontsize=11)
        ax.set_ylim(0, max(max(rates) if rates else 0, baseline) * 1.2 + 0.05)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.2)

    ax3.set_xlabel("Layer", fontsize=12)
    ax3.set_xticks(layers[::4])
    ax3.set_xlim(-0.5, max(layers) + 0.5)

    fig.suptitle(
        f"{model_name} — Aggregate patch histogram [{pos_label}] (5 pairs, N=100)\n"
        f"Corrupt→Clean: corrupt rhyme rate on clean run",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_model(model_cfg):
    name       = model_cfg["name"]
    agg_dir    = os.path.join(RESULTS_DIR, model_cfg["agg_dir"])
    fear_dir   = os.path.join(RESULTS_DIR, model_cfg["fear_subdir"])
    agg_json   = os.path.join(agg_dir, "aggregate.json")
    out_prefix = model_cfg["out_prefix"]
    out_dir    = agg_dir

    print(f"\n=== {name} ===")

    PAIR_IDS = ["doom_dread", "bliss_joy", "dark_night", "grief_pain"]

    # Load data for all positions
    all_combined       = {}
    all_combined_clean = {}
    all_baselines      = {}
    all_baselines_clean = {}

    for pos_id in POSITIONS:
        # corrupt rates (for existing overlay/subplot plots)
        agg_rates  = load_aggregate_rates(agg_json, pos_id)
        fear_c, fear_r = load_fear_both_rates(fear_dir, pos_id, N_FEAR)
        combined   = compute_combined(agg_rates, fear_r)

        # clean rates (needed for histograms)
        pair_clean_all = []
        pair_corrupt_all = []
        for pair_id in PAIR_IDS:
            pc, pr = load_pair_rates(agg_dir, pair_id, pos_id)
            pair_clean_all.append(pc)
            pair_corrupt_all.append(pr)
        # mean across 4 new pairs + fear
        n_layers_pos = len(fear_c)
        combined_clean = [
            (sum(pair_clean_all[p][l] for p in range(4)) + fear_c[l]) / 5
            for l in range(n_layers_pos)
        ]

        # baselines
        agg_base  = load_aggregate_baseline(agg_dir, pos_id)
        fear_base = load_fear_baseline(fear_dir, pos_id, N_FEAR)
        baseline  = (agg_base * 4 + fear_base) / 5

        # clean baseline: average baseline_clean_rate from pair_meta + fear
        pair_clean_bases = []
        for pair_id in PAIR_IDS:
            with open(os.path.join(agg_dir, pair_id, "pair_meta.json")) as f:
                meta = json.load(f)
            pair_clean_bases.append(meta["baseline_clean_rate"])
        fear_clean_base = sum(fear_c) / len(fear_c) if fear_c else 0  # rough; use actual baseline
        # actually load fear clean baseline properly
        exp_id = FEAR_DIR_MAP[pos_id]
        candidates = [d for d in os.listdir(fear_dir) if exp_id in d]
        run_dir = os.path.join(fear_dir, sorted(candidates)[0])
        with open(os.path.join(run_dir, "generations.json")) as f:
            fear_data = json.load(f)
        fear_clean_base = fear_data["baseline"].get("unpatched_clean_clean_rhyme_rate", 0.0)
        baseline_clean = (sum(pair_clean_bases) + fear_clean_base) / 5

        all_combined[pos_id]       = combined
        all_combined_clean[pos_id] = combined_clean
        all_baselines[pos_id]      = baseline
        all_baselines_clean[pos_id] = baseline_clean

        n_layers = len(combined)
        best_layer = int(np.argmax(combined))
        print(f"  {pos_id}: best layer={best_layer} ({combined[best_layer]:.3f}), baseline={baseline:.3f}")

    n_layers = len(list(all_combined.values())[0])
    layers   = list(range(n_layers))

    # ── Plot 1: 6 subplots, one per position ────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    axes = axes.flatten()

    for ax, pos_id, pos_label in zip(axes, POSITIONS, POS_LABELS):
        combined = all_combined[pos_id]
        baseline = all_baselines[pos_id]
        ax.plot(layers, combined, color="steelblue", linewidth=1.5)
        ax.axhline(baseline, color="gray", linestyle="--", linewidth=1, label=f"baseline ({baseline:.3f})")
        ax.set_title(pos_label, fontsize=13)
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel("Mean corrupt rhyme rate", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{name} — Aggregate (5 pairs, N=100)\nCorrupt→Clean patching: corrupt rhyme rate on clean run", fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{out_prefix}_aggregate_subplots.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # ── Plot 2: all positions overlaid ──────────────────────────────────────────
    colors = cm.tab10(np.linspace(0, 0.6, len(POSITIONS)))
    fig, ax = plt.subplots(figsize=(12, 5))

    for (pos_id, pos_label), color in zip(zip(POSITIONS, POS_LABELS), colors):
        combined = all_combined[pos_id]
        label = pos_label.replace("\n", " ")
        ax.plot(layers, combined, color=color, linewidth=1.5, label=label)

    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("Mean corrupt rhyme rate", fontsize=13)
    ax.set_title(f"{name} — All positions overlaid (5 pairs, N=100)", fontsize=13)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{out_prefix}_aggregate_overlay.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # ── Plot 3: per-position histograms ──────────────────────────────────────────
    for pos_id, pos_label in zip(POSITIONS, POS_LABELS):
        label = pos_label.replace("\n", " ")
        out_path = os.path.join(out_dir, f"{out_prefix}_aggregate_{pos_id}_histogram.png")
        plot_histogram(
            layers         = list(range(len(all_combined[pos_id]))),
            clean_rates    = all_combined_clean[pos_id],
            corrupt_rates  = all_combined[pos_id],
            baseline_clean  = all_baselines_clean[pos_id],
            baseline_corrupt = all_baselines[pos_id],
            pos_label      = label,
            model_name     = name,
            out_path       = out_path,
        )


# -- Main ------------------------------------------------------------------------

if __name__ == "__main__":
    for model_cfg in MODELS:
        plot_model(model_cfg)
    print("\nDone.")
