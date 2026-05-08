"""
Aggregate path-patch K-sweep results from both workers.

Reads per-cell JSONs from results/gemma3_27b_path_patch_sweep/worker{0,1}/,
computes mean ± SD of corrupt_rhyme_rate per (K, ranking), and produces:
  - aggregate.json with full breakdown
  - path_patch_curve.png with K-sweep curves for attnweight, commactrl, and random
"""

import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results/gemma3_27b_path_patch_sweep")
OUT_PNG = os.path.join(ROOT, "../icml2026/media/results-patching/path_patch_curve.png")
OUT_JSON = os.path.join(RESULTS, "aggregate.json")

# Load all cells
cells = []
for w in ("worker0", "worker1"):
    for path in glob.glob(os.path.join(RESULTS, w, "K*.json")):
        with open(path) as f:
            cells.append(json.load(f))
print(f"Loaded {len(cells)} cells")

# Group by (K, ranking, seed) → list of pair-level corrupt_rhyme_rate
groups = defaultdict(list)
for c in cells:
    key = (c["K"], c["ranking"], c["seed"])
    groups[key].append(c["corrupt_rhyme_rate"])

# For attnweight + commactrl: 1 number per K (mean over 5 pairs)
# For random: 5 seeds, each is mean over 5 pairs, then take mean ± SD across seeds
def stats(values):
    a = np.array(values)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1) if len(a) > 1 else 0.0),
            "n": len(a), "values": a.tolist()}

result = {}
K_values = sorted({k for k, _, _ in groups.keys()})
for K in K_values:
    result[K] = {}
    # attnweight: 1 group, 5 pair values → mean
    aw = groups.get((K, "attnweight", None), [])
    if aw:
        result[K]["attnweight"] = stats(aw)
    # commactrl
    cc = groups.get((K, "commactrl", None), [])
    if cc:
        result[K]["commactrl"] = stats(cc)
    # random: collect each seed's pair-mean then average
    seed_means = []
    for (kk, r, s), vals in groups.items():
        if kk == K and r == "random":
            seed_means.append(np.mean(vals))
    if seed_means:
        result[K]["random"] = {"mean": float(np.mean(seed_means)),
                                "std":  float(np.std(seed_means, ddof=1) if len(seed_means) > 1 else 0.0),
                                "n_seeds": len(seed_means),
                                "per_seed_mean": [float(x) for x in seed_means]}

# Write aggregate JSON
with open(OUT_JSON, "w") as f:
    json.dump({"K_values": K_values, "by_K": {str(k): result[k] for k in K_values}},
              f, indent=2)

# Print summary table
print(f"\n{'K':<4} {'attnweight':<14} {'commactrl':<12} {'random (5 seeds)':<24}")
print("-" * 60)
for K in K_values:
    aw = result[K].get("attnweight", {})
    cc = result[K].get("commactrl", {})
    rd = result[K].get("random", {})
    aw_s = f"{aw.get('mean', float('nan')):.3f}"
    cc_s = f"{cc.get('mean', float('nan')):.3f}"
    rd_s = f"{rd.get('mean', float('nan')):.3f} ± {rd.get('std', 0):.3f}"
    print(f"{K:<4} {aw_s:<14} {cc_s:<12} {rd_s}")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
xs = np.array(K_values)
aw_means = np.array([result[K]["attnweight"]["mean"] for K in K_values])
cc_means = np.array([result[K]["commactrl"]["mean"]  for K in K_values])
rd_means = np.array([result[K]["random"]["mean"]     for K in K_values])
rd_stds  = np.array([result[K]["random"]["std"]      for K in K_values])

ax.plot(xs, aw_means, marker="o", linewidth=2, color="#2255cc",
        label="attention-weight ranking (top-K)")
ax.plot(xs, cc_means, marker="s", linewidth=2, color="#999999",
        label="comma-control (negative)")
ax.errorbar(xs, rd_means, yerr=rd_stds, marker="^", linewidth=2, color="#cc6677",
             label="random (5 seeds, ±SD)", capsize=4)

ax.set_xlabel("K (number of heads patched at i=0)", fontsize=13)
ax.set_ylabel("Corrupt rhyme rate (mean over 5 pairs × N=20)", fontsize=13)
ax.set_xticks(xs)
ax.set_ylim(-0.02, 1.0)
ax.set_title("Gemma-3-27B: K-sweep at i=0 with controls", fontsize=13)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(loc="upper left", frameon=False, fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
print(f"\nSaved {OUT_PNG}")
