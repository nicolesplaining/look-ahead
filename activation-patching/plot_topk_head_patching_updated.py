"""
Updated Fig. 7b: top-k head patching corrupt-rhyme rate with cluster-bootstrap
95% CIs over prompt pairs, plus a CI'd reference line for the full-residual
peak and a CI on the headline "fraction of full effect" ratio.

Outputs:
  - icml2026/media/results-patching/topk_head_patching-updated.png

Inputs:
  - activation-patching/results/gemma3_27b_topk_head_patching/results.json
        (5 pairs × {k=1,2,3,5,10}, per-pair corrupt_rate)
  - activation-patching/results/GEMMA3_AGGREGATE/gemma3_27b_aggregate_N20/aggregate.json
        (4 pairs at i=0 across all layers, per-pair rates)
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


def cluster_bootstrap(per_pair, n_boot=N_BOOT, alpha=0.05, seed=RNG_SEED):
    """per_pair: 1-D array of per-pair rates. Returns (mean, lo, hi)."""
    per_pair = np.asarray(per_pair, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(per_pair)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = per_pair[idx].mean(axis=1)
    return float(per_pair.mean()), float(np.percentile(boot, 100*alpha/2)), float(np.percentile(boot, 100*(1-alpha/2)))


def joint_cluster_bootstrap_ratio(num_per_pair, den_per_pair, n_boot=N_BOOT, alpha=0.05, seed=RNG_SEED):
    """
    For ratio mean(num)/mean(den), resample SHARED pair indices in both.
    Pair lists must be aligned (same pair order, same length).
    """
    num = np.asarray(num_per_pair, dtype=float)
    den = np.asarray(den_per_pair, dtype=float)
    assert len(num) == len(den)
    rng = np.random.default_rng(seed)
    n = len(num)
    idx = rng.integers(0, n, size=(n_boot, n))
    num_b = num[idx].mean(axis=1)
    den_b = den[idx].mean(axis=1)
    # guard against zeros in denominator (shouldn't happen at the peak, but be safe)
    safe = den_b > 1e-9
    ratios = num_b[safe] / den_b[safe]
    point = num.mean() / den.mean()
    return float(point), float(np.percentile(ratios, 100*alpha/2)), float(np.percentile(ratios, 100*(1-alpha/2)))


# ── Load top-k head patching ───────────────────────────────────────────────────
HEAD_JSON = os.path.join(RES, "gemma3_27b_topk_head_patching", "results.json")
with open(HEAD_JSON) as f:
    head = json.load(f)
head_pairs = [pr["pair_id"] for pr in head["pair_results"]]
k_values = head["k_values"]
# matrix shape (n_k, n_pairs) of per-pair corrupt rates
head_mat = np.zeros((len(k_values), len(head_pairs)))
for j, pr in enumerate(head["pair_results"]):
    for i, k in enumerate(k_values):
        rec = next(r for r in pr["k_results"] if r["k"] == k)
        head_mat[i, j] = rec["corrupt_rate"]

# ── Load full-residual i=0 peak from GEMMA3_AGGREGATE (4 pairs, N=20) ─────────
FULL_JSON = os.path.join(RES, "GEMMA3_AGGREGATE", "gemma3_27b_aggregate_N20", "aggregate.json")
with open(FULL_JSON) as f:
    full = json.load(f)
agg_pairs = full["pairs"]
i0_layers = full["aggregate"]["i_0"]
# find peak layer by mean
means = [r["mean_corrupt_rhyme_rate"] for r in i0_layers]
peak_layer = int(np.argmax(means))
peak_entry = i0_layers[peak_layer]
agg_full = {p: peak_entry["per_pair_corrupt_rhyme_rate"][p] for p in agg_pairs}

# fright_fear was run separately at N=100; take first N=20 to match the other
# four pairs' sample size (otherwise the 5-pair mean is asymmetrically weighted).
from plot_main_per_layer import load_fear_corrupt_rate_per_layer
fear_rates = load_fear_corrupt_rate_per_layer(
    os.path.join(RES, "GEMMA3_PER_LAYER"), "i_0"
)
agg_full["fright_fear"] = fear_rates[peak_layer]

# Build the 5-pair full-residual rate vector aligned to head_pairs
full_per_pair_at_peak = np.array([agg_full[p] for p in head_pairs], dtype=float)
full_pairs = head_pairs[:]
full_mean, full_lo, full_hi = cluster_bootstrap(full_per_pair_at_peak)
print(
    f"Full-residual peak: layer {peak_layer}, "
    f"5-pair mean={full_mean:.3f} CI [{full_lo:.3f}, {full_hi:.3f}], "
    f"per-pair={dict(zip(head_pairs, full_per_pair_at_peak.round(3)))}"
)
# also report the 4-pair-only mean so we can cross-check the paper's reference
m4 = float(np.mean([agg_full[p] for p in agg_pairs]))
print(f"  (paper-style 4-pair full mean at L{peak_layer} = {m4:.3f})")

# ── Per-k CI ───────────────────────────────────────────────────────────────────
ks_means, ks_lo, ks_hi = [], [], []
for i, k in enumerate(k_values):
    m, lo, hi = cluster_bootstrap(head_mat[i])
    ks_means.append(m); ks_lo.append(lo); ks_hi.append(hi)
    print(f"  k={k:2d}: mean={m:.3f} CI [{lo:.3f}, {hi:.3f}]  (n_pairs={len(head_pairs)})")

# ── Headline ratio: k=5 / full-residual peak (5 pairs, joint cluster bootstrap) ─
k5_idx = k_values.index(5)
num_per_pair = head_mat[k5_idx]                       # aligned to head_pairs
den_per_pair = full_per_pair_at_peak                  # aligned to head_pairs
ratio, ratio_lo, ratio_hi = joint_cluster_bootstrap_ratio(num_per_pair, den_per_pair)
print(
    f"k=5 / full-residual peak ratio (5-pair joint bootstrap): "
    f"{ratio:.3f} CI [{ratio_lo:.3f}, {ratio_hi:.3f}]"
)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))

x = np.arange(len(k_values))
ks_means = np.array(ks_means); ks_lo = np.array(ks_lo); ks_hi = np.array(ks_hi)
err = np.vstack([ks_means - ks_lo, ks_hi - ks_means])

ax.bar(x, ks_means, color="#6c8ebf", width=0.55,
       yerr=err, error_kw=dict(ecolor="black", lw=1.0, capsize=3))

# Reference line + shaded band for full-residual peak CI
ax.axhline(full_mean, linestyle="--", color="gray", linewidth=1.4,
           label=f"Full residual stream (best layer) = {full_mean:.2f}")
ax.fill_between([-0.6, len(k_values) - 0.4], full_lo, full_hi,
                color="gray", alpha=0.15, linewidth=0)

ax.set_xlim(-0.6, len(k_values) - 0.4)
ax.set_xticks(x)
ax.set_xticklabels([f"k={k}" for k in k_values], fontsize=12)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Corrupt rhyme rate", fontsize=14)
ax.tick_params(axis="y", labelsize=12)
ax.legend(loc="upper left", frameon=False, fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out_path = os.path.join(OUT, "topk_head_patching-updated.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {out_path}")
