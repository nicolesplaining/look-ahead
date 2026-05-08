"""Aggregate two-stage path-patch results, plot side-by-side with single-stage."""

import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_2S = os.path.join(ROOT, "results/gemma3_27b_path_patch_two_stage")
RESULTS_1S = os.path.join(ROOT, "results/gemma3_27b_path_patch_sweep")
OUT_PNG = os.path.join(ROOT, "../icml2026/media/results-patching/path_patch_two_stage_curve.png")
OUT_PNG_SIDE = os.path.join(ROOT, "../icml2026/media/results-patching/path_patch_compare.png")
OUT_JSON = os.path.join(RESULTS_2S, "aggregate.json")


def load_cells(results_dir):
    cells = []
    for w in ("worker0", "worker1"):
        for path in glob.glob(os.path.join(results_dir, w, "K*.json")):
            with open(path) as f:
                cells.append(json.load(f))
    return cells


def boot_ci(values, B=10000, seed=0):
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    boots = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(B)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)

def aggregate(cells):
    groups = defaultdict(list)
    for c in cells:
        groups[(c["K"], c["ranking"], c["seed"])].append(c["corrupt_rhyme_rate"])
    K_values = sorted({k for k, _, _ in groups.keys()})
    out = {}
    for K in K_values:
        out[K] = {}
        aw = groups.get((K, "attnweight", None), [])
        if aw:
            m, lo, hi = boot_ci(aw)
            out[K]["attnweight"] = {"mean": m, "ci_lo": lo, "ci_hi": hi, "values": aw, "n": len(aw)}
        cc = groups.get((K, "commactrl", None), [])
        if cc:
            m, lo, hi = boot_ci(cc)
            out[K]["commactrl"] = {"mean": m, "ci_lo": lo, "ci_hi": hi, "values": cc, "n": len(cc)}
        seed_means = []
        for (kk, r, s), vals in groups.items():
            if kk == K and r == "random":
                seed_means.append(float(np.mean(vals)))
        if seed_means:
            out[K]["random"] = {
                "mean": float(np.mean(seed_means)),
                "std": float(np.std(seed_means, ddof=1) if len(seed_means) > 1 else 0.0),
                "per_seed_mean": seed_means,
            }
    return K_values, out


# Load both
cells_2s = load_cells(RESULTS_2S)
cells_1s = load_cells(RESULTS_1S)
print(f"two-stage cells: {len(cells_2s)}")
print(f"single-stage cells: {len(cells_1s)}")

K_values, agg_2s = aggregate(cells_2s)
_,        agg_1s = aggregate(cells_1s)

# Save two-stage aggregate
with open(OUT_JSON, "w") as f:
    json.dump({"K_values": K_values,
                "by_K": {str(k): agg_2s[k] for k in K_values}},
               f, indent=2)

# Print comparison table
print(f"\n{'K':<4} {'2S aw':<10} {'1S aw':<10} {'2S cc':<8} {'1S cc':<8} {'2S rand':<14} {'1S rand':<14}")
print("-" * 75)
for K in K_values:
    aw2 = agg_2s[K].get("attnweight", {}).get("mean", float("nan"))
    aw1 = agg_1s.get(K, {}).get("attnweight", {}).get("mean", float("nan"))
    cc2 = agg_2s[K].get("commactrl", {}).get("mean", float("nan"))
    cc1 = agg_1s.get(K, {}).get("commactrl", {}).get("mean", float("nan"))
    r2  = agg_2s[K].get("random", {})
    r1  = agg_1s.get(K, {}).get("random", {})
    r2s = f"{r2.get('mean', float('nan')):.3f}±{r2.get('std', 0):.3f}"
    r1s = f"{r1.get('mean', float('nan')):.3f}±{r1.get('std', 0):.3f}" if r1 else "—"
    print(f"{K:<4} {aw2:<10.3f} {aw1:<10.3f} {cc2:<8.3f} {cc1:<8.3f} {r2s:<14} {r1s:<14}")

# Plot 1: two-stage only
xs = np.array(K_values)
def get_col(agg, key, sub="mean"):
    return np.array([agg.get(K, {}).get(key, {}).get(sub, np.nan) for K in K_values])

fig, ax = plt.subplots(figsize=(8, 2.7))
aw_means = get_col(agg_2s, "attnweight")
aw_lo    = get_col(agg_2s, "attnweight", "ci_lo")
aw_hi    = get_col(agg_2s, "attnweight", "ci_hi")
ax.errorbar(xs, aw_means, yerr=[aw_means - aw_lo, aw_hi - aw_means],
            marker="o", linewidth=2, color="#2255cc", capsize=3,
            label="attention-weight top-K")
ax.plot(xs, get_col(agg_2s, "commactrl"), marker="s", linewidth=2, color="#999999",
        label="comma-control")
ax.errorbar(xs, get_col(agg_2s, "random"),
            yerr=get_col(agg_2s, "random", "std"),
            marker="^", linewidth=2, color="#cc6677", capsize=3,
            label="random")
ax.set_xlabel("K", fontsize=11)
ax.set_ylabel("Mean corrupt rhyme rate", fontsize=11)
ax.set_xticks(xs); ax.set_ylim(-0.02, 1.0)
ax.tick_params(labelsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(loc="upper right", frameon=False, fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
plt.close()
print(f"\nSaved {OUT_PNG}")

# Plot 2: side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, (label, agg) in zip(axes, [("Single-stage (paper's method)", agg_1s),
                                     ("Two-stage (Wang et al. style)", agg_2s)]):
    ax.plot(xs, get_col(agg, "attnweight"), marker="o", linewidth=2,
            color="#2255cc", label="attention-weight (top-K)")
    ax.plot(xs, get_col(agg, "commactrl"), marker="s", linewidth=2,
            color="#999999", label="comma-control")
    ax.errorbar(xs, get_col(agg, "random"),
                yerr=get_col(agg, "random", "std"),
                marker="^", linewidth=2, color="#cc6677", capsize=4,
                label="random (5 seeds, ±SD)")
    ax.set_xlabel("K", fontsize=13)
    ax.set_xticks(xs); ax.set_ylim(-0.02, 1.0)
    ax.set_title(label, fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
axes[0].set_ylabel("Corrupt rhyme rate", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_PNG_SIDE, dpi=200, bbox_inches="tight")
print(f"Saved {OUT_PNG_SIDE}")
