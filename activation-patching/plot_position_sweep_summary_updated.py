"""
Compact summary of the full position sweep (Figs. 16-19): for each model size,
report the peak-layer corrupt rhyme rate with a cluster-bootstrap 95% CI for
each of the 6 patched positions. Output as a single TSV.
"""

import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "..", "icml2026", "media", "results-patching")

N_BOOT = 10_000
RNG_SEED = 0


def cluster_bootstrap(per_pair, n_boot=N_BOOT, alpha=0.05, seed=RNG_SEED):
    per_pair = np.asarray(per_pair, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(per_pair)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = per_pair[idx].mean(axis=1)
    return float(per_pair.mean()), float(np.percentile(boot, 100*alpha/2)), float(np.percentile(boot, 100*(1-alpha/2)))


MODELS = [
    # (family, label, aggregate.json path)
    ("Qwen3", "0.6B", os.path.join(RES, "AGGREGATE", "Qwen_Qwen3-0.6B_aggregate_N20", "aggregate.json")),
    ("Qwen3", "1.7B", os.path.join(RES, "AGGREGATE", "Qwen_Qwen3-1.7B_aggregate_N20", "aggregate.json")),
    ("Qwen3", "4B",   os.path.join(RES, "AGGREGATE", "Qwen_Qwen3-4B_aggregate_N20",   "aggregate.json")),
    ("Qwen3", "8B",   os.path.join(RES, "AGGREGATE", "Qwen_Qwen3-8B_aggregate_N20",   "aggregate.json")),
    ("Qwen3", "14B",  os.path.join(RES, "AGGREGATE", "Qwen_Qwen3-14B_aggregate_N20",  "aggregate.json")),
    ("Qwen3", "32B",  os.path.join(RES, "QWEN3_AGGREGATE", "qwen3_32b_aggregate_N20", "aggregate.json")),
    ("Gemma-3", "1B",  os.path.join(RES, "AGGREGATE", "google_gemma-3-1b-it_aggregate_N20",  "aggregate.json")),
    ("Gemma-3", "4B",  os.path.join(RES, "AGGREGATE", "google_gemma-3-4b-it_aggregate_N20",  "aggregate.json")),
    ("Gemma-3", "12B", os.path.join(RES, "AGGREGATE", "google_gemma-3-12b-it_aggregate_N20", "aggregate.json")),
    ("Gemma-3", "27B", os.path.join(RES, "GEMMA3_AGGREGATE", "gemma3_27b_aggregate_N20",     "aggregate.json")),
    ("Llama-3", "1B",  os.path.join(RES, "AGGREGATE", "meta-llama_Llama-3.2-1B-Instruct_aggregate_N20", "aggregate.json")),
    ("Llama-3", "3B",  os.path.join(RES, "AGGREGATE", "meta-llama_Llama-3.2-3B-Instruct_aggregate_N20", "aggregate.json")),
    ("Llama-3", "8B",  os.path.join(RES, "AGGREGATE", "meta-llama_Llama-3.1-8B-Instruct_aggregate_N20", "aggregate.json")),
]

POSITIONS = ["i_minus2", "i_minus1", "i_0", "i_plus1", "i_plus2", "i_plus3"]


def peak_per_pair_at_pos(agg_dict, pos):
    """Returns per-pair rates at the layer with maximum mean rate."""
    layers = agg_dict[pos]
    means = [r["mean_corrupt_rhyme_rate"] for r in layers]
    peak_L = int(np.argmax(means))
    pair_rates = layers[peak_L]["per_pair_corrupt_rhyme_rate"]
    pairs = list(pair_rates.keys())
    return peak_L, np.array([pair_rates[p] for p in pairs])


def main():
    rows = ["Family\tSize\t" + "\t".join(POSITIONS) + "\tPeak position"]

    for fam, sz, agg_path in MODELS:
        if not os.path.exists(agg_path):
            rows.append(f"{fam}\t{sz}\t" + "\t".join(["—"]*len(POSITIONS)) + "\t—")
            continue
        with open(agg_path) as f:
            d = json.load(f)
        agg = d["aggregate"]
        cells = []
        peak_pos_mean = -1.0
        peak_pos_name = ""
        for pos in POSITIONS:
            if pos not in agg:
                cells.append("—"); continue
            peak_L, per_pair = peak_per_pair_at_pos(agg, pos)
            m, lo, hi = cluster_bootstrap(per_pair)
            cells.append(f"L{peak_L}: {m:.2f} [{lo:.2f}, {hi:.2f}]")
            if m > peak_pos_mean:
                peak_pos_mean = m
                peak_pos_name = pos
        rows.append(f"{fam}\t{sz}\t" + "\t".join(cells) + f"\t{peak_pos_name}")

    out = os.path.join(OUT, "position_sweep_summary-updated.tsv")
    with open(out, "w") as f:
        f.write("\n".join(rows) + "\n")
    print("\n".join(rows))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
