"""
Adds CIs to the remaining figures/tables that were not yet covered:
  - Fig. 1 (Pile look-ahead probes, top-5):  pile-{qwen,gemma,llama}-top5-updated.png
  - Fig. 8 (top-1 appendix):                 pile-{qwen,gemma}-top1-updated.png
  - Fig. 10 (patching baselines):            patching_baselines-updated.png
  - top-k MLP patching (text only in paper): topk_mlp_patching-updated.png
  - Table 3 (all-layers patching):           all_layers_patching_table-updated.tsv

Also writes summary lines to stdout — peak rates, CI widths, and any cells where
the CI changes the qualitative reading.
"""

import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
RES_PATCH = os.path.join(ROOT, "results")
RES_PROBE = os.path.join(ROOT, "..", "probe", "results")
OUT_PATCH = os.path.join(ROOT, "..", "icml2026", "media", "results-patching")
OUT_PROBE = os.path.join(ROOT, "..", "icml2026", "media", "results-probe")

Z = 1.959963984540054
N_BOOT = 10_000
RNG_SEED = 0


def wilson_ci(p, n, z=Z):
    p = np.asarray(p, dtype=float)
    n = np.asarray(n, dtype=float)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return centre - half, centre + half


def cluster_bootstrap(per_pair, n_boot=N_BOOT, alpha=0.05, seed=RNG_SEED):
    per_pair = np.asarray(per_pair, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(per_pair)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = per_pair[idx].mean(axis=1)
    return float(per_pair.mean()), float(np.percentile(boot, 100*alpha/2)), float(np.percentile(boot, 100*(1-alpha/2)))


# ──────────────────────────────────────────────────────────────────────────────
# 1) Pile look-ahead probes (Fig. 1, Fig. 8)
# ──────────────────────────────────────────────────────────────────────────────
def compute_pile_n_per_k(tokens_jsonl, k_values):
    """For each k: N = sum over val sequences of (total_len - k)."""
    lengths = []
    with open(tokens_jsonl) as f:
        for line in f:
            lengths.append(len(json.loads(line)["tokens"]))
    lengths = np.array(lengths)
    return {k: int((lengths - k).clip(min=0).sum()) for k in k_values}


PILE_PANELS = [
    # (out_basename, model_dir, paper_label)
    ("pile-qwen",  "Qwen3-32B",                "Qwen3-32B"),
    ("pile-gemma", "Gemma-3-27B",              "Gemma-3-27B"),
    ("pile-llama", "Llama-3.1-70B-Instruct",   "Llama-3.1-70B"),
]

K_LIST = [1, 2, 3, 8]
K_COLORS = {1: "#2c5fa8", 2: "#5891d6", 3: "#9bc1e8", 8: "#cccccc"}
K_LSTYLES = {1: "-", 2: "-", 3: "-", 8: "-"}


def load_pile_curve(model_dir, k, metric_key):
    p = os.path.join(RES_PROBE, model_dir, "experiment_results_linear", "experiment_results.json")
    if not os.path.exists(p):
        return None, None
    with open(p) as f:
        d = json.load(f)
    pairs = []
    for _, v in d["results"].items():
        if v.get("k") == k and v.get(metric_key) is not None:
            pairs.append((v["layer"], v[metric_key]))
    pairs.sort()
    return np.array([l for l, _ in pairs]), np.array([v for _, v in pairs])


def load_unigram(model_dir, k, metric_key):
    p = os.path.join(RES_PROBE, model_dir, "baselines", "unigram_results.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        d = json.load(f)
    for _, v in d["results"].items():
        if v.get("k") == k and v.get(metric_key) is not None:
            return float(v[metric_key])
    return None


def plot_pile(metric_short):
    """metric_short: 'top5' or 'top1'."""
    metric_key = "val_top5_accuracy" if metric_short == "top5" else "val_accuracy"
    n_per_k = compute_pile_n_per_k(
        os.path.join(ROOT, "..", "probe", "data", "activations_val.tokens.jsonl"),
        K_LIST,
    )
    print(f"  Pile {metric_short}: N per k = {n_per_k}")

    for out_base, model_dir, title in PILE_PANELS:
        if metric_short == "top1" and out_base == "pile-llama":
            # paper Fig. 8 only shows Qwen and Gemma top1 (and Llama isn't required)
            continue
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        for k in K_LIST:
            layers, vals = load_pile_curve(model_dir, k, metric_key)
            if layers is None:
                continue
            lo, hi = wilson_ci(vals, n_per_k[k])
            color = K_COLORS[k]
            ax.fill_between(layers, lo, hi, color=color, alpha=0.18, linewidth=0)
            ax.plot(layers, vals, label=f"k={k}", color=color, lw=2.0,
                    linestyle=K_LSTYLES[k])
        # unigram dashed line + CI band
        u_k1 = load_unigram(model_dir, 1, metric_key)
        if u_k1 is not None:
            n_unigram = n_per_k[1]
            u_lo, u_hi = wilson_ci(u_k1, n_unigram)
            ax.axhline(u_k1, linestyle="--", lw=1.4, color="#999999", label="Unigram")
            ax.axhspan(u_lo, u_hi, color="#cccccc", alpha=0.3, lw=0)
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_ylim(0, 0.7 if metric_short == "top5" else 0.5)
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, ncols=2, loc="upper left")
        plt.tight_layout()
        out = os.path.join(OUT_PROBE, f"{out_base}-{metric_short}-updated.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out}")


# ──────────────────────────────────────────────────────────────────────────────
# 2) Patching baselines (Fig. 10)
# ──────────────────────────────────────────────────────────────────────────────
BASELINE_JOBS = [
    # (out_basename, true_per_layer_dir, baseline_root, last_word_pos_id, paper_label)
    ("baselines_qwen3_32b",
     os.path.join(RES_PATCH, "QWEN3_PER_LAYER", "qwen3_32b_exp_fear_minus1_corrupt_to_clean", "generations.json"),
     os.path.join(RES_PATCH, "QWEN3_BASELINE"),
     "i_minus1", "Qwen3-32B"),
    ("baselines_gemma3_27b",
     os.path.join(RES_PATCH, "GEMMA3_PER_LAYER", "gemma3_27b_exp_fear_minus2_corrupt_to_clean", "generations.json"),
     os.path.join(RES_PATCH, "GEMMA3_BASELINE"),
     "i_minus2", "Gemma-3-27B"),
]


def per_layer_avg_across_pairs(baseline_root, baseline_kind):
    """Pool over pair sub-dirs. Returns (n_layers,) array of pooled rates and N."""
    pair_dirs = [d for d in sorted(os.listdir(baseline_root))
                 if os.path.isdir(os.path.join(baseline_root, d))]
    rates_per_layer = None
    n_per_layer = None
    for pair in pair_dirs:
        gpath = os.path.join(baseline_root, pair, baseline_kind, "generations.json")
        if not os.path.exists(gpath):
            continue
        with open(gpath) as f:
            d = json.load(f)
        n_layers = len(d["results"])
        if rates_per_layer is None:
            rates_per_layer = np.zeros(n_layers)
            n_per_layer = np.zeros(n_layers, dtype=int)
        sn = d["sampling_n"]
        for L, r in enumerate(d["results"]):
            rates_per_layer[L] += r["corrupt_rhyme_rate"] * sn   # successes
            n_per_layer[L] += sn
    return rates_per_layer / np.maximum(n_per_layer, 1), n_per_layer


def plot_baselines():
    for out_base, true_path, baseline_root, lw_pos, title in BASELINE_JOBS:
        # true patching curve at lw_pos: from PER_LAYER experiment with N=100/layer
        with open(true_path) as f:
            true_d = json.load(f)
        n_layers = len(true_d["results"])
        true_rates = np.array([r["corrupt_rhyme_rate"] for r in true_d["results"]])
        N_true = true_d["sampling_n"]    # per layer
        true_lo, true_hi = wilson_ci(true_rates, N_true)

        zero_rates, zero_N = per_layer_avg_across_pairs(baseline_root, "zero_vector")
        donor_rates, donor_N = per_layer_avg_across_pairs(baseline_root, "donor_prompt")
        zero_lo, zero_hi = wilson_ci(zero_rates, zero_N.clip(min=1))
        donor_lo, donor_hi = wilson_ci(donor_rates, donor_N.clip(min=1))

        layers = np.arange(n_layers)
        fig, ax = plt.subplots(figsize=(7, 3.6))

        # True
        ax.fill_between(layers, true_lo, true_hi, color="#cc6677", alpha=0.18, lw=0)
        ax.plot(layers, true_rates, color="#cc6677", lw=2.0, label="Activation patching")
        # Zero-vector
        ax.fill_between(layers, zero_lo, zero_hi, color="#888888", alpha=0.18, lw=0)
        ax.plot(layers, zero_rates, color="#888888", lw=1.6, linestyle="--",
                label="Zero-vector baseline")
        # Donor-prompt
        ax.fill_between(layers, donor_lo, donor_hi, color="#4477aa", alpha=0.18, lw=0)
        ax.plot(layers, donor_rates, color="#4477aa", lw=1.6, linestyle=":",
                label="Donor-prompt baseline")

        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Corrupt rhyme rate", fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"({title}) — patching at last word", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="upper right", frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        out = os.path.join(OUT_PATCH, f"{out_base}-updated.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out}  N(true)={N_true}, N(baseline-pooled)≈{int(zero_N[0])}")
        # Also report max-CI-upper-bound for baselines (defends "near zero" claim)
        print(f"    {title}  zero-vector max upper-CI = {zero_hi.max():.3f}; "
              f"donor-prompt max upper-CI = {donor_hi.max():.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# 3) Top-k MLP patching (text-only claim: "zero across all k")
# ──────────────────────────────────────────────────────────────────────────────
def plot_topk_mlp():
    p = os.path.join(RES_PATCH, "gemma3_27b_topk_mlp_patching", "results.json")
    with open(p) as f:
        d = json.load(f)
    k_values = d["k_values"]
    pairs = [pr["pair_id"] for pr in d["pair_results"]]
    mat = np.zeros((len(k_values), len(pairs)))
    for j, pr in enumerate(d["pair_results"]):
        for i, k in enumerate(k_values):
            rec = next(r for r in pr["k_results"] if r["k"] == k)
            mat[i, j] = rec["corrupt_rate"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    means, los, his = [], [], []
    pooled_n = d["sampling_n"] * len(pairs)
    for i, k in enumerate(k_values):
        m, lo, hi = cluster_bootstrap(mat[i])
        means.append(m); los.append(lo); his.append(hi)
        # also Wilson on pooled
        wlo, whi = wilson_ci(np.array([m]), pooled_n)
        print(f"  k={k:2d}: mean={m:.3f} cluster-CI [{lo:.3f},{hi:.3f}] Wilson(N={pooled_n}) [{float(wlo):.3f},{float(whi):.3f}]")
    means = np.array(means); los = np.array(los); his = np.array(his)
    err = np.vstack([means - los, his - means])
    x = np.arange(len(k_values))
    ax.bar(x, means, color="#888888", width=0.55,
           yerr=err, error_kw=dict(ecolor="black", lw=1.0, capsize=3))
    ax.set_xticks(x); ax.set_xticklabels([f"k={k}" for k in k_values])
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("Corrupt rhyme rate")
    ax.set_title("Top-k MLP patching at newline (Gemma-3-27B)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(OUT_PATCH, "topk_mlp_patching-updated.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ──────────────────────────────────────────────────────────────────────────────
# 4) Table 3 (all-layers patching) — write a TSV with Wilson 95% CIs
# ──────────────────────────────────────────────────────────────────────────────
TABLE3_MODELS = [
    # (family, label, dir-name in ALL_LAYERS_AGGREGATE)
    ("Qwen3", "0.6B", "Qwen_Qwen3-0.6B_all_layers_N20"),
    ("Qwen3", "1.7B", "Qwen_Qwen3-1.7B_all_layers_N20"),
    ("Qwen3", "4B",   "Qwen_Qwen3-4B_all_layers_N20"),
    ("Qwen3", "8B",   "Qwen_Qwen3-8B_all_layers_N20"),
    ("Qwen3", "14B",  "Qwen_Qwen3-14B_all_layers_N20"),
    ("Qwen3", "32B",  None),  # comes from QWEN3_AGGREGATE
    ("Gemma-3", "1B",  "google_gemma-3-1b-it_all_layers_N20"),
    ("Gemma-3", "4B",  "google_gemma-3-4b-it_all_layers_N20"),
    ("Gemma-3", "12B", "google_gemma-3-12b-it_all_layers_N20"),
    ("Gemma-3", "27B", None),
    ("Llama-3", "1B",  "meta-llama_Llama-3.2-1B-Instruct_all_layers_N20"),
    ("Llama-3", "3B",  "meta-llama_Llama-3.2-3B-Instruct_all_layers_N20"),
    ("Llama-3", "8B",  "meta-llama_Llama-3.1-8B-Instruct_all_layers_N20"),
    ("Llama-3", "70B", None),
]


def _wilson_scalar(p, n):
    lo, hi = wilson_ci(np.array([p]), n)
    return float(lo[0]), float(hi[0])


def _read_aggregate(agg_path, position):
    """Returns (mean_rate, n_pairs) or (None,None)."""
    if not os.path.exists(agg_path):
        return None, None
    with open(agg_path) as f:
        d = json.load(f)
    agg = d.get("aggregate")
    if isinstance(agg, dict) and position in agg and isinstance(agg[position], dict):
        return float(agg[position]["mean_corrupt_rhyme_rate"]), len(d["pairs"])
    return None, None


def _read_largest_model_run(genfile, lw_offset, nw_offset=0):
    """Single-prompt all-layers runs (N=100). Returns ((lw_rate, lw_N), (nw_rate, nw_N))."""
    if not os.path.exists(genfile):
        return (None, None), (None, None)
    with open(genfile) as f:
        d = json.load(f)
    n = d["sampling_n"]
    lw = nw = None
    for pr in d["position_results"]:
        off = pr.get("position_offset")
        if off == lw_offset: lw = pr["corrupt_rhyme_rate"]
        if off == nw_offset: nw = pr["corrupt_rhyme_rate"]
    return (lw, n), (nw, n)


def write_table3():
    rows = ["Family\tSize\tLastWord %\tLW 95% CI\tNewline %\tNL 95% CI\tN\tNotes"]

    def emit(fam, sz, lw, lw_n, nw, nw_n, note=""):
        if lw is None:
            lw_str, lw_ci = "—", "—"
        else:
            lo, hi = _wilson_scalar(lw, lw_n)
            lw_str = f"{int(round(lw*100))}"
            lw_ci = f"[{int(round(lo*100))},{int(round(hi*100))}]"
        if nw is None:
            nw_str, nw_ci = "—", "—"
        else:
            lo, hi = _wilson_scalar(nw, nw_n)
            nw_str = f"{int(round(nw*100))}"
            nw_ci = f"[{int(round(lo*100))},{int(round(hi*100))}]"
        N = lw_n or nw_n or "—"
        rows.append(f"{fam}\t{sz}\t{lw_str}\t{lw_ci}\t{nw_str}\t{nw_ci}\t{N}\t{note}")

    # Smaller models — 5 pairs × N=20 (pooled N=100) via ALL_LAYERS_AGGREGATE
    SMALLER = [
        ("Qwen3", "0.6B", "Qwen_Qwen3-0.6B_all_layers_N20", "i_minus1"),
        ("Qwen3", "1.7B", "Qwen_Qwen3-1.7B_all_layers_N20", "i_minus1"),
        ("Qwen3", "4B",   "Qwen_Qwen3-4B_all_layers_N20",   "i_minus1"),
        ("Qwen3", "8B",   "Qwen_Qwen3-8B_all_layers_N20",   "i_minus1"),
        ("Qwen3", "14B",  "Qwen_Qwen3-14B_all_layers_N20",  "i_minus1"),
        ("Gemma-3", "1B",  "google_gemma-3-1b-it_all_layers_N20",  "i_minus2"),
        ("Gemma-3", "4B",  "google_gemma-3-4b-it_all_layers_N20",  "i_minus2"),
        ("Gemma-3", "12B", "google_gemma-3-12b-it_all_layers_N20", "i_minus2"),
        ("Llama-3", "1B", "meta-llama_Llama-3.2-1B-Instruct_all_layers_N20", "i_minus1"),
        ("Llama-3", "3B", "meta-llama_Llama-3.2-3B-Instruct_all_layers_N20", "i_minus1"),
        ("Llama-3", "8B", "meta-llama_Llama-3.1-8B-Instruct_all_layers_N20", "i_minus1"),
    ]
    for fam, sz, dirname, lw_pos in SMALLER:
        agg = os.path.join(RES_PATCH, "ALL_LAYERS_AGGREGATE", dirname, "aggregate.json")
        lw, npairs = _read_aggregate(agg, lw_pos)
        nw, _ = _read_aggregate(agg, "i_0")
        N = (npairs or 5) * 20
        emit(fam, sz, lw, N, nw, N, note=f"{npairs} pairs × N=20")

    # Largest models — single fright/fear run, N=100
    qwen32 = os.path.join(RES_PATCH, "Qwen3-32B-fright-fear-all-layers", "generations.json")
    (lw, lw_n), (nw, nw_n) = _read_largest_model_run(qwen32, lw_offset=-1, nw_offset=0)
    emit("Qwen3", "32B", lw, lw_n, nw, nw_n, note="single fright/fear prompt, N=100")

    gemma27 = os.path.join(RES_PATCH, "gemma3_27b_patch_all_layers_N100_v3", "generations.json")
    (lw, lw_n), (nw, nw_n) = _read_largest_model_run(gemma27, lw_offset=-2, nw_offset=0)
    emit("Gemma-3", "27B", lw, lw_n, nw, nw_n, note="single fright/fear prompt, N=100")

    # Llama-70B — 5 pairs × N=20 at i=-1 only; no i=0 data captured in all-layers run
    llama_base = os.path.join(RES_PATCH, "llama-3.1-70b-all-layers")
    if os.path.exists(llama_base):
        rates = []
        for pair in sorted(os.listdir(llama_base)):
            gp = os.path.join(llama_base, pair, "i_minus1", "generations.json")
            if os.path.exists(gp):
                rates.append(json.load(open(gp))["corrupt_rhyme_rate"])
        if rates:
            lw = float(np.mean(rates)); lw_n = 20 * len(rates)
            emit("Llama-3", "70B", lw, lw_n, None, None,
                 note=f"{len(rates)} pairs × N=20; i=0 not in all-layers run")
        else:
            emit("Llama-3", "70B", None, None, None, None, note="missing")

    out = os.path.join(OUT_PATCH, "all_layers_patching_table-updated.tsv")
    with open(out, "w") as f:
        f.write("\n".join(rows) + "\n")
    print("\n".join(rows))
    print(f"\n  Saved {out}")


def main():
    print("== Pile probe top-5 (Fig. 1) ==")
    plot_pile("top5")
    print("\n== Pile probe top-1 (Fig. 8) ==")
    plot_pile("top1")
    print("\n== Patching baselines (Fig. 10) ==")
    plot_baselines()
    print("\n== Top-k MLP patching ==")
    plot_topk_mlp()
    print("\n== Table 3 (all-layers patching) ==")
    write_table3()


if __name__ == "__main__":
    main()
