"""
Updated probe figures for Sec. 3.2 of the paper:

  - Fig. 3 a/b/c (top-5 accuracy):     <model>-top5-updated.png
  - Fig. 3 d/e/f (top-5 rhyme accuracy): <model>-rhyme1-updated.png
  - Fig. 4 (scaling):                   scaling-top5-updated.png, scaling-rhyme1-updated.png

What's added:
  * Fig. 3: Wilson 95% CI shaded band per (layer, position). N=200 val items.
  * Fig. 4: error bars on each model's "max-layer gap" between i=0 and i=1.
            CI on the gap = ±1.96 * SE, where
                SE^2 = (p0(1-p0) + p1(1-p1)) / N
            (treats the two probes' val correctness as independent — we don't have
             per-sample data, so we cannot exploit the pairing. This is a conservative
             upper bound on the real SE.)
            Caveat noted in stats_analysis_added.md: "max over layers" is biased
            upward; the bar gives the optimistic point estimate.
"""

import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "..", "results")
OUT = os.path.join(ROOT, "..", "..", "icml2026", "media", "results-poem")

N_VAL = 200          # validation items per (layer, i) — see metadata in JSONs
Z = 1.959963984540054  # 95% normal quantile


# ── Wilson 95% CI ─────────────────────────────────────────────────────────────
def wilson_ci(p, n=N_VAL, z=Z):
    """Wilson score interval for a single proportion. p, n scalar or array-like."""
    p = np.asarray(p, dtype=float)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return centre - half, centre + half


# ── Fig. 3 line plots with shaded CIs ─────────────────────────────────────────
PROBE_PANELS = [
    # (model_dir, panel_label_for_title, list of i_keys)
    ("qwen3-32B",                "Qwen3-32B",      ["i_neg1", "i0", "i1", "i2", "i3"]),
    ("Gemma-3-27B",              "Gemma-3-27B",    ["i_neg2", "i_neg1", "i0", "i1", "i2", "i3"]),
    ("Llama-3.1-70B-Instruct",   "Llama-3.1-70B",  ["i_neg1", "i0", "i1", "i2", "i3"]),
]

# colours / styles consistent with poem/scripts/plot_results.sh
I_STYLES = {
    "i_neg2": dict(label="i=-2", color="#F4D03F", ls="-",  lw=2.0),
    "i_neg1": dict(label="i=-1", color="#F4D03F", ls="-",  lw=2.0),
    "i0":     dict(label="i=0",  color="tomato",  ls="-",  lw=2.2),
    "i1":     dict(label="i=1",  color="#93C4E0", ls="--", lw=1.8),
    "i2":     dict(label="i=2",  color="#6AAFD4", ls="--", lw=1.8),
    "i3":     dict(label="i=3",  color="#4195C3", ls="--", lw=1.8),
}

METRIC_KEYS = {
    "top5":   "val_top5_accuracy",
    "rhyme1": "top5_rhyme_accuracy",   # paper Fig. 3 d/e/f is Rhyme@5
}


def model_to_paper_key(model_dir):
    if "qwen" in model_dir.lower():  return "qwen"
    if "gemma" in model_dir.lower(): return "gemma"
    if "llama" in model_dir.lower(): return "llama"
    raise ValueError(model_dir)


def load_layer_metric(model_dir, i_key, metric_key):
    path = os.path.join(RES, model_dir, i_key, "experiment_results.json")
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        d = json.load(f)
    pairs = []
    for _k, v in d["results"].items():
        if metric_key in v and v[metric_key] is not None:
            pairs.append((v["layer"], v[metric_key]))
    pairs.sort()
    layers = np.array([l for l, _ in pairs])
    vals = np.array([v for _, v in pairs])
    return layers, vals


def plot_fig3(metric_short):
    metric_key = METRIC_KEYS[metric_short]
    for model_dir, _title, i_keys in PROBE_PANELS:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for i_key in i_keys:
            layers, vals = load_layer_metric(model_dir, i_key, metric_key)
            if layers is None:
                continue
            lo, hi = wilson_ci(vals)
            style = I_STYLES[i_key]
            ax.fill_between(layers, lo, hi, color=style["color"], alpha=0.18, linewidth=0)
            ax.plot(layers, vals, label=style["label"], color=style["color"],
                    linestyle=style["ls"], linewidth=style["lw"])
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, loc="upper left", ncols=2)
        plt.tight_layout()
        out_name = f"{model_to_paper_key(model_dir)}-{metric_short}-updated.png"
        plt.savefig(os.path.join(OUT, out_name), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_name}")


# ── Fig. 4 scaling bars with paired-gap error bars ────────────────────────────
SCALING_MODELS = [
    # (family, label, dir)
    ("Gemma-3", "1B",  "Gemma-3-1B"),
    ("Gemma-3", "4B",  "Gemma-3-4B"),
    ("Gemma-3", "12B", "Gemma-3-12B"),
    ("Gemma-3", "27B", "Gemma-3-27B"),
    ("Qwen3",   "0.6B", "Qwen3-0.6B"),
    ("Qwen3",   "1.7B", "Qwen3-1.7B"),
    ("Qwen3",   "4B",   "Qwen3-4B"),
    ("Qwen3",   "8B",   "Qwen3-8B"),
    ("Qwen3",   "14B",  "Qwen3-14B"),
    ("Qwen3",   "32B",  "qwen3-32B"),
    ("Llama-3.1/3.2", "1B",  "Llama-3.2-1B"),
    ("Llama-3.1/3.2", "3B",  "Llama-3.2-3B"),
    ("Llama-3.1/3.2", "8B",  "Llama-3.1-8B"),
    ("Llama-3.1/3.2", "70B", "Llama-3.1-70B-Instruct"),
]
FAMILY_COLORS = {
    "Gemma-3":         "#4477aa",
    "Qwen3":           "#cc6677",
    "Llama-3.1/3.2":   "#117733",
}


def gap_with_ci(model_dir, metric_key, n=N_VAL):
    """
    Returns (max_gap, ci_lo, ci_hi, peak_layer, p0_at_peak, p1_at_peak).
    Gap is a0_layer - a1_layer; CI uses unpaired-Wald approximation since per-sample
    correctness isn't saved (gives a conservative — wider — interval than paired).
    """
    layers0, a0 = load_layer_metric(model_dir, "i0", metric_key)
    layers1, a1 = load_layer_metric(model_dir, "i1", metric_key)
    if layers0 is None or layers1 is None:
        return None
    n_min = min(len(a0), len(a1))
    a0, a1 = a0[:n_min], a1[:n_min]
    gaps = a0 - a1
    peak = int(np.argmax(gaps))
    p0, p1 = float(a0[peak]), float(a1[peak])
    se = math.sqrt(max(p0 * (1 - p0) + p1 * (1 - p1), 0) / n)
    half = Z * se
    return float(gaps[peak]), float(gaps[peak] - half), float(gaps[peak] + half), peak, p0, p1


def plot_fig4(metric_short, ylabel="Accuracy difference"):
    metric_key = METRIC_KEYS[metric_short]
    fam_data = {}
    for fam, lab, mdir in SCALING_MODELS:
        info = gap_with_ci(mdir, metric_key)
        if info is None:
            print(f"  SKIP {fam} {lab} ({mdir})")
            continue
        gap, lo, hi, layer, p0, p1 = info
        fam_data.setdefault(fam, []).append((lab, gap, lo, hi, layer, p0, p1))
        print(f"  {fam:<15} {lab:>5}  L{layer:>2}  i0={p0:.3f}  i1={p1:.3f}  gap={gap:.3f} CI[{lo:.3f},{hi:.3f}]")

    n_fams = len(fam_data)
    fig, axes = plt.subplots(1, n_fams, figsize=(4 * n_fams, 3.2), sharey=True)
    if n_fams == 1:
        axes = [axes]
    for ax, (fam, items) in zip(axes, fam_data.items()):
        labs   = [it[0] for it in items]
        gaps   = np.array([it[1] for it in items])
        los    = np.array([it[2] for it in items])
        his    = np.array([it[3] for it in items])
        x = np.arange(len(items))
        err = np.vstack([gaps - los, his - gaps])
        # clip negative lower bars at 0 visually but keep CI length truthful
        err_low_clipped = np.minimum(err[0], gaps)
        err_disp = np.vstack([err_low_clipped, err[1]])
        ax.bar(x, gaps, color=FAMILY_COLORS[fam], width=0.6,
               yerr=err_disp,
               error_kw=dict(ecolor="black", lw=1.0, capsize=3))
        ax.set_xticks(x)
        ax.set_xticklabels(labs, fontsize=10)
        ax.set_title(fam, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axhline(0, color="black", lw=0.5)
    plt.tight_layout()
    out_name = f"scaling-{metric_short}-updated.png"
    plt.savefig(os.path.join(OUT, out_name), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_name}\n")


def main():
    print("== Fig. 3 (line plots with Wilson 95% CI bands) ==")
    plot_fig3("top5")
    plot_fig3("rhyme1")
    print("\n== Fig. 4 (scaling bars with gap CIs) ==")
    plot_fig4("top5")
    plot_fig4("rhyme1")


if __name__ == "__main__":
    main()
