"""
Updated version of plot_main_per_layer.py: same per-layer bar charts, but with
cluster-bootstrap 95% CIs over prompt pairs added as black error bars.

The unit of independence is the prompt pair (4 pairs for Qwen/Gemma, 5 for Llama).
We resample pairs with replacement (10,000 reps), recompute the mean corrupt-rhyme
rate per layer, and report the 2.5/97.5 percentiles.

Outputs (next to the originals, with -updated suffix):
  - patching_qwen3_32b-updated.png
  - patching_gemma3_27b-updated.png
  - patching_llama_70b-updated.png
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "..", "icml2026", "media", "results-patching")

N_BOOT = 10_000
RNG_SEED = 0


def load_per_pair_canonical(agg_path, pos_keys):
    """For Qwen/Gemma aggregate.json: per-pair rates are inlined per layer."""
    with open(agg_path) as f:
        d = json.load(f)
    pairs = d["pairs"]
    n_layers = len(d["aggregate"][pos_keys[0]])
    out = {}
    for pos in pos_keys:
        # shape (n_layers, n_pairs)
        mat = np.zeros((n_layers, len(pairs)))
        for L, entry in enumerate(d["aggregate"][pos]):
            for j, p in enumerate(pairs):
                mat[L, j] = entry["per_pair_corrupt_rhyme_rate"][p]
        out[pos] = mat
    return out, pairs


def load_per_pair_llama(base_dir, pos_keys):
    """Llama per-layer-per-position: walk per-pair generations.json files."""
    pairs = sorted(
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    )
    # peek to learn n_layers
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


def cluster_bootstrap_ci(per_pair_mat, n_boot=N_BOOT, alpha=0.05, seed=RNG_SEED):
    """
    per_pair_mat: shape (n_layers, n_pairs), each entry = per-pair corrupt rate.
    Returns means, lo, hi: each shape (n_layers,).
    """
    rng = np.random.default_rng(seed)
    n_layers, n_pairs = per_pair_mat.shape
    idx = rng.integers(0, n_pairs, size=(n_boot, n_pairs))   # resample pair indices
    # boot[b, L] = mean over resampled pairs at layer L
    boot = per_pair_mat[:, idx].mean(axis=2).T               # (n_boot, n_layers)
    means = per_pair_mat.mean(axis=1)
    lo = np.percentile(boot, 100 * alpha / 2, axis=0)
    hi = np.percentile(boot, 100 * (1 - alpha / 2), axis=0)
    return means, lo, hi


JOBS = [
    {
        "out": "patching_qwen3_32b-updated.png",
        "loader": "canonical",
        "agg": f"{RES}/QWEN3_AGGREGATE/qwen3_32b_aggregate_N20/aggregate.json",
        "lw_pos": "i_minus1", "lw_label": "i=-1",
    },
    {
        "out": "patching_gemma3_27b-updated.png",
        "loader": "canonical",
        "agg": f"{RES}/GEMMA3_AGGREGATE/gemma3_27b_aggregate_N20/aggregate.json",
        "lw_pos": "i_minus2", "lw_label": "i=-2",
    },
    {
        "out": "patching_llama_70b-updated.png",
        "loader": "llama",
        "agg": f"{RES}/llama-3.1-70b-per-layer-per-position",
        "lw_pos": "i_minus1", "lw_label": "i=-1",
    },
]


def main():
    for job in JOBS:
        pos_keys = [job["lw_pos"], "i_0"]
        if job["loader"] == "canonical":
            mats, pairs = load_per_pair_canonical(job["agg"], pos_keys)
        else:
            mats, pairs = load_per_pair_llama(job["agg"], pos_keys)

        m_lw, lo_lw, hi_lw = cluster_bootstrap_ci(mats[job["lw_pos"]])
        m_i0, lo_i0, hi_i0 = cluster_bootstrap_ci(mats["i_0"])

        n = len(m_lw)
        x = np.arange(n)
        w = 0.4

        fig, ax = plt.subplots(figsize=(8, 4))
        # asymmetric error bars
        err_lw = np.vstack([m_lw - lo_lw, hi_lw - m_lw])
        err_i0 = np.vstack([m_i0 - lo_i0, hi_i0 - m_i0])

        ax.bar(
            x - w / 2, m_lw, w,
            yerr=err_lw, label=job["lw_label"], color="#6699cc",
            error_kw=dict(ecolor="black", lw=0.6, capsize=1.2, alpha=0.85),
        )
        ax.bar(
            x + w / 2, m_i0, w,
            yerr=err_i0, label="i=0", color="#cc6677",
            error_kw=dict(ecolor="black", lw=0.6, capsize=1.2, alpha=0.85),
        )
        ax.set_xlabel("Layer", fontsize=18)
        ax.set_ylabel("Corrupt rhyme rate", fontsize=18)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-1, n)
        step = max(1, n // 16)
        ax.set_xticks(np.arange(0, n, step))
        ax.tick_params(axis="both", labelsize=14)
        ax.legend(loc="upper right", frameon=False, fontsize=16)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        out_path = os.path.join(OUT, job["out"])
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

        peak_lw = int(np.argmax(m_lw))
        peak_i0 = int(np.argmax(m_i0))
        print(
            f"{job['out']}  pairs={pairs}  n_pairs={len(pairs)}  "
            f"peak {job['lw_label']}: L{peak_lw} = {m_lw[peak_lw]:.3f} "
            f"[{lo_lw[peak_lw]:.3f}, {hi_lw[peak_lw]:.3f}]  "
            f"peak i=0: L{peak_i0} = {m_i0[peak_i0]:.3f} "
            f"[{lo_i0[peak_i0]:.3f}, {hi_i0[peak_i0]:.3f}]"
        )


if __name__ == "__main__":
    main()
