"""
Updated appendix per-layer patching figures with cluster-bootstrap 95% CI error
bars on each layer's bars. Same panels and layout as the originals; the only
addition is uncertainty.

Outputs:
  appendix_patching_Qwen3_small-updated.png   (0.6B, 1.7B, 4B)
  appendix_patching_Qwen3_large-updated.png   (8B, 14B, 32B)
  appendix_patching_Gemma3-updated.png        (1B, 4B, 12B, 27B)
  appendix_patching_Llama3-updated.png        (1B, 3B, 8B, 70B)
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
OUT_DIR = os.path.join(ROOT, "..", "icml2026", "media", "results-patching")

N_BOOT = 10_000
RNG_SEED = 0


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
    sample = json.load(open(os.path.join(base_dir, pairs[0], pos_keys[0], "generations.json")))
    n_layers = sample["n_layers"]
    out = {pos: np.zeros((n_layers, len(pairs))) for pos in pos_keys}
    for j, pair in enumerate(pairs):
        for pos in pos_keys:
            with open(os.path.join(base_dir, pair, pos, "generations.json")) as f:
                gen = json.load(f)
            for L, r in enumerate(gen["results"]):
                out[pos][L, j] = r["corrupt_rhyme_rate"]
    return out, pairs


def cluster_bootstrap_ci(per_pair_mat, n_boot=N_BOOT, alpha=0.05, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n_layers, n_pairs = per_pair_mat.shape
    idx = rng.integers(0, n_pairs, size=(n_boot, n_pairs))
    boot = per_pair_mat[:, idx].mean(axis=2).T
    means = per_pair_mat.mean(axis=1)
    lo = np.percentile(boot, 100 * alpha / 2, axis=0)
    hi = np.percentile(boot, 100 * (1 - alpha / 2), axis=0)
    return means, lo, hi


def make_figure(jobs, lw_pos, lw_label, out_name, panel_height=2.2):
    fig, axes = plt.subplots(len(jobs), 1, figsize=(10, panel_height * len(jobs)), sharey=True)
    if len(jobs) == 1:
        axes = [axes]

    for ax, job in zip(axes, jobs):
        if job["fmt"] == "canonical":
            mats, pairs = load_per_pair_canonical(job["path"], [lw_pos, "i_0"])
        else:
            mats, pairs = load_per_pair_llama(job["path"], [lw_pos, "i_0"])
        m_lw, lo_lw, hi_lw = cluster_bootstrap_ci(mats[lw_pos])
        m_i0, lo_i0, hi_i0 = cluster_bootstrap_ci(mats["i_0"])

        n = len(m_lw)
        x = np.arange(n)
        w = 0.4
        err_lw = np.vstack([m_lw - lo_lw, hi_lw - m_lw])
        err_i0 = np.vstack([m_i0 - lo_i0, hi_i0 - m_i0])

        ax.bar(
            x - w/2, m_lw, w, yerr=err_lw, label=lw_label, color="#6699cc",
            error_kw=dict(ecolor="black", lw=0.5, capsize=0.8, alpha=0.85),
        )
        ax.bar(
            x + w/2, m_i0, w, yerr=err_i0, label="i=0", color="#cc6677",
            error_kw=dict(ecolor="black", lw=0.5, capsize=0.8, alpha=0.85),
        )
        ax.set_title(job["title"], fontsize=10, loc="left")
        ax.set_ylabel("Corrupt rhyme rate", fontsize=9)
        ax.set_ylim(0, 1.10)
        ax.set_xlim(-1, n)
        step = max(1, n // 20)
        ax.set_xticks(np.arange(0, n, step))
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        peak_lw = int(np.argmax(m_lw))
        peak_i0 = int(np.argmax(m_i0))
        print(
            f"  {job['title']:<32}  n_pairs={len(pairs)}  "
            f"peak {lw_label}: L{peak_lw} = {m_lw[peak_lw]:.2f} "
            f"[{lo_lw[peak_lw]:.2f}, {hi_lw[peak_lw]:.2f}]  "
            f"peak i=0: L{peak_i0} = {m_i0[peak_i0]:.2f} "
            f"[{lo_i0[peak_i0]:.2f}, {hi_i0[peak_i0]:.2f}]"
        )

    axes[-1].set_xlabel("Layer", fontsize=10)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, out_name)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}\n")


def main():
    print("=== Qwen3 small (0.6B, 1.7B, 4B) ===")
    make_figure(
        [
            {"title": "Qwen3-0.6B", "fmt": "canonical",
             "path": f"{RES}/AGGREGATE/Qwen_Qwen3-0.6B_aggregate_N20/aggregate.json"},
            {"title": "Qwen3-1.7B", "fmt": "canonical",
             "path": f"{RES}/AGGREGATE/Qwen_Qwen3-1.7B_aggregate_N20/aggregate.json"},
            {"title": "Qwen3-4B",   "fmt": "canonical",
             "path": f"{RES}/AGGREGATE/Qwen_Qwen3-4B_aggregate_N20/aggregate.json"},
        ],
        lw_pos="i_minus1", lw_label="i=-1",
        out_name="appendix_patching_Qwen3_small-updated.png",
    )

    print("=== Qwen3 large (8B, 14B, 32B) ===")
    make_figure(
        [
            {"title": "Qwen3-8B",  "fmt": "canonical",
             "path": f"{RES}/AGGREGATE/Qwen_Qwen3-8B_aggregate_N20/aggregate.json"},
            {"title": "Qwen3-14B", "fmt": "canonical",
             "path": f"{RES}/AGGREGATE/Qwen_Qwen3-14B_aggregate_N20/aggregate.json"},
            {"title": "Qwen3-32B", "fmt": "canonical",
             "path": f"{RES}/QWEN3_AGGREGATE/qwen3_32b_aggregate_N20/aggregate.json"},
        ],
        lw_pos="i_minus1", lw_label="i=-1",
        out_name="appendix_patching_Qwen3_large-updated.png",
    )

    print("=== Gemma-3 (1B, 4B, 12B, 27B) ===")
    make_figure(
        [
            {"title": "Gemma-3-1B",  "fmt": "canonical",
             "path": f"{RES}/AGGREGATE/google_gemma-3-1b-it_aggregate_N20/aggregate.json"},
            {"title": "Gemma-3-4B",  "fmt": "canonical",
             "path": f"{RES}/AGGREGATE/google_gemma-3-4b-it_aggregate_N20/aggregate.json"},
            {"title": "Gemma-3-12B", "fmt": "canonical",
             "path": f"{RES}/AGGREGATE/google_gemma-3-12b-it_aggregate_N20/aggregate.json"},
            {"title": "Gemma-3-27B", "fmt": "canonical",
             "path": f"{RES}/GEMMA3_AGGREGATE/gemma3_27b_aggregate_N20/aggregate.json"},
        ],
        lw_pos="i_minus2", lw_label="i=-2",
        out_name="appendix_patching_Gemma3-updated.png",
    )

    print("=== Llama-3 (1B, 3B, 8B, 70B) ===")
    make_figure(
        [
            {"title": "Llama-3.2-1B-Instruct",  "fmt": "canonical",
             "path": f"{RES}/AGGREGATE/meta-llama_Llama-3.2-1B-Instruct_aggregate_N20/aggregate.json"},
            {"title": "Llama-3.2-3B-Instruct",  "fmt": "canonical",
             "path": f"{RES}/AGGREGATE/meta-llama_Llama-3.2-3B-Instruct_aggregate_N20/aggregate.json"},
            {"title": "Llama-3.1-8B-Instruct",  "fmt": "canonical",
             "path": f"{RES}/AGGREGATE/meta-llama_Llama-3.1-8B-Instruct_aggregate_N20/aggregate.json"},
            {"title": "Llama-3.1-70B-Instruct", "fmt": "llama",
             "path": f"{RES}/llama-3.1-70b-per-layer-per-position"},
        ],
        lw_pos="i_minus1", lw_label="i=-1",
        out_name="appendix_patching_Llama3-updated.png",
    )


if __name__ == "__main__":
    main()
