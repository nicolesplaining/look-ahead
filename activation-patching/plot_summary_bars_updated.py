"""
Updated patching_summary_bars: peak corrupt rhyme rate per (model, position) with
cluster-bootstrap 95% CI error bars over prompt pairs.

For each model, we load the per-pair matrix (layers x pairs) at last-word and i=0,
pick the peak layer (max mean), and cluster-bootstrap the peak-layer mean over
pair indices (10,000 reps, seed=0).

Caveat: peak-layer is selected from the same data, so the point estimate is
upward-biased. The CI is for the chosen-layer mean, not the population max.

Output:
  icml2026/media/results-patching/patching_summary_bars-updated.png
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
OUT_PATH = os.path.join(
    ROOT, "..", "icml2026", "media", "results-patching",
    "patching_summary_bars-updated.png",
)

N_BOOT = 10_000
RNG_SEED = 0


# (family, label, agg_path, last_word_pos, format)
MODELS = [
    ("Qwen3", "0.6B", f"{RES}/AGGREGATE/Qwen_Qwen3-0.6B_aggregate_N20/aggregate.json", "i_minus1", "canonical"),
    ("Qwen3", "1.7B", f"{RES}/AGGREGATE/Qwen_Qwen3-1.7B_aggregate_N20/aggregate.json", "i_minus1", "canonical"),
    ("Qwen3", "4B",   f"{RES}/AGGREGATE/Qwen_Qwen3-4B_aggregate_N20/aggregate.json",   "i_minus1", "canonical"),
    ("Qwen3", "8B",   f"{RES}/AGGREGATE/Qwen_Qwen3-8B_aggregate_N20/aggregate.json",   "i_minus1", "canonical"),
    ("Qwen3", "14B",  f"{RES}/AGGREGATE/Qwen_Qwen3-14B_aggregate_N20/aggregate.json",  "i_minus1", "canonical"),
    ("Qwen3", "32B",  f"{RES}/QWEN3_AGGREGATE/qwen3_32b_aggregate_N20/aggregate.json", "i_minus1", "canonical"),
    ("Gemma-3", "1B",  f"{RES}/AGGREGATE/google_gemma-3-1b-it_aggregate_N20/aggregate.json",  "i_minus2", "canonical"),
    ("Gemma-3", "4B",  f"{RES}/AGGREGATE/google_gemma-3-4b-it_aggregate_N20/aggregate.json",  "i_minus2", "canonical"),
    ("Gemma-3", "12B", f"{RES}/AGGREGATE/google_gemma-3-12b-it_aggregate_N20/aggregate.json", "i_minus2", "canonical"),
    ("Gemma-3", "27B", f"{RES}/GEMMA3_AGGREGATE/gemma3_27b_aggregate_N20/aggregate.json",     "i_minus2", "canonical"),
    ("Llama-3", "1B",  f"{RES}/AGGREGATE/meta-llama_Llama-3.2-1B-Instruct_aggregate_N20/aggregate.json", "i_minus1", "canonical"),
    ("Llama-3", "3B",  f"{RES}/AGGREGATE/meta-llama_Llama-3.2-3B-Instruct_aggregate_N20/aggregate.json", "i_minus1", "canonical"),
    ("Llama-3", "8B",  f"{RES}/AGGREGATE/meta-llama_Llama-3.1-8B-Instruct_aggregate_N20/aggregate.json", "i_minus1", "canonical"),
    ("Llama-3", "70B", f"{RES}/llama-3.1-70b-per-layer-per-position",                                    "i_minus1", "llama"),
]


def load_per_pair_canonical(agg_path, pos_keys):
    with open(agg_path) as f:
        d = json.load(f)
    pairs = d["pairs"]
    n_layers = len(d["aggregate"][pos_keys[0]])
    out = {}
    for pos in pos_keys:
        mat = np.zeros((n_layers, len(pairs)))
        for L, entry in enumerate(d["aggregate"][pos]):
            for j, p in enumerate(pairs):
                mat[L, j] = entry["per_pair_corrupt_rhyme_rate"][p]
        out[pos] = mat
    return out, pairs


def load_per_pair_llama(base_dir, pos_keys):
    pairs = sorted(
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    )
    sample_path = os.path.join(base_dir, pairs[0], pos_keys[0], "generations.json")
    with open(sample_path) as f:
        n_layers = json.load(f)["n_layers"]
    out = {pos: np.zeros((n_layers, len(pairs))) for pos in pos_keys}
    for j, pair in enumerate(pairs):
        for pos in pos_keys:
            path = os.path.join(base_dir, pair, pos, "generations.json")
            with open(path) as f:
                gen = json.load(f)
            for L, r in enumerate(gen["results"]):
                out[pos][L, j] = r["corrupt_rhyme_rate"]
    return out, pairs


def peak_with_ci(per_pair_mat, n_boot=N_BOOT, alpha=0.05, seed=RNG_SEED):
    """Select peak layer (argmax of mean), then bootstrap pairs at that layer."""
    means = per_pair_mat.mean(axis=1)
    peak_L = int(np.argmax(means))
    pair_rates = per_pair_mat[peak_L]
    rng = np.random.default_rng(seed)
    n = len(pair_rates)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = pair_rates[idx].mean(axis=1)
    return peak_L, float(means[peak_L]), float(np.percentile(boot, 100*alpha/2)), float(np.percentile(boot, 100*(1-alpha/2)))


def main():
    rows = []
    for fam, sz, path, lw_pos, fmt in MODELS:
        if fmt == "canonical":
            mats, pairs = load_per_pair_canonical(path, [lw_pos, "i_0"])
        else:
            mats, pairs = load_per_pair_llama(path, [lw_pos, "i_0"])
        L_lw, m_lw, lo_lw, hi_lw = peak_with_ci(mats[lw_pos])
        L_i0, m_i0, lo_i0, hi_i0 = peak_with_ci(mats["i_0"])
        rows.append({
            "fam": fam, "sz": sz, "n_pairs": len(pairs),
            "lw": (m_lw, lo_lw, hi_lw, L_lw),
            "i0": (m_i0, lo_i0, hi_i0, L_i0),
        })
        print(
            f"{fam:>8} {sz:>5}  n_pairs={len(pairs)}  "
            f"lw L{L_lw}={m_lw:.3f} [{lo_lw:.3f},{hi_lw:.3f}]   "
            f"i0 L{L_i0}={m_i0:.3f} [{lo_i0:.3f},{hi_i0:.3f}]"
        )

    fig, ax = plt.subplots(figsize=(11, 3.8))
    x = np.arange(len(rows))
    w = 0.4
    lw_means = np.array([r["lw"][0] for r in rows])
    lw_lo    = np.array([r["lw"][1] for r in rows])
    lw_hi    = np.array([r["lw"][2] for r in rows])
    i0_means = np.array([r["i0"][0] for r in rows])
    i0_lo    = np.array([r["i0"][1] for r in rows])
    i0_hi    = np.array([r["i0"][2] for r in rows])
    err_lw = np.vstack([lw_means - lw_lo, lw_hi - lw_means])
    err_i0 = np.vstack([i0_means - i0_lo, i0_hi - i0_means])

    ax.bar(
        x - w/2, lw_means, w, yerr=err_lw, label="last word",
        color="#6699cc",
        error_kw=dict(ecolor="black", lw=0.7, capsize=1.8, alpha=0.85),
    )
    ax.bar(
        x + w/2, i0_means, w, yerr=err_i0, label="i=0 (newline)",
        color="#cc6677",
        error_kw=dict(ecolor="black", lw=0.7, capsize=1.8, alpha=0.85),
    )

    ax.set_xticks(x)
    ax.set_xticklabels([r["sz"] for r in rows], fontsize=9)

    families = [r["fam"] for r in rows]
    boundaries = []
    prev = None
    for i, fam in enumerate(families):
        if fam != prev:
            boundaries.append(i)
        prev = fam
    boundaries.append(len(families))
    for j in range(len(boundaries) - 1):
        s, e = boundaries[j], boundaries[j + 1]
        fam = families[s]
        mid = (s + e - 1) / 2
        ax.text(mid, 1.07, fam, ha="center", va="bottom", fontsize=10,
                transform=ax.get_xaxis_transform())
        if j > 0:
            ax.axvline(s - 0.5, color="gray", linestyle=":", alpha=0.5)

    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Peak corrupt rhyme rate")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
