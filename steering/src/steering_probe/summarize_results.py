#!/usr/bin/env python3
"""
Aggregate results.json -> summary.json.

Averages steered_rhyme_pct across all (src, tgt) pairs for each
(layer, position), producing a flat list of descriptive records.

Usage:
    python -m steering_probe.summarize_results \
        --results-path steering/results/results.json \
        --output-path  steering/results/summary.json
"""
import argparse
import json
import os
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser(description="Summarize steering results")
    p.add_argument("--results-path", required=True, help="Path to results.json")
    p.add_argument("--output-path", required=True, help="Path to write summary.json")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.results_path) as f:
        data = json.load(f)

    # Structure: data[src][tgt][layer][position] -> {steered_rhyme_pct, ...}
    # Average steered_rhyme_pct across (src, tgt) pairs for each (layer, position).
    totals = defaultdict(lambda: {"sum": 0.0, "count": 0})

    for src, targets in data.items():
        for tgt, layers in targets.items():
            for layer, positions in layers.items():
                for pos, leaf in positions.items():
                    key = (int(layer), int(pos))
                    totals[key]["sum"] += leaf["steered_rhyme_pct"]
                    totals[key]["count"] += 1

    records = [
        {
            "layer": layer,
            "position": position,
            "steered_rhyme_pct": round(v["sum"] / v["count"], 4),
            "n_pairs_averaged": v["count"],
        }
        for (layer, position), v in sorted(totals.items())
    ]

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(records, f, indent=2)

    nonzero = sum(1 for r in records if r["steered_rhyme_pct"] > 0)
    print(f"Written {len(records)} records ({nonzero} non-zero) -> {args.output_path}")


if __name__ == "__main__":
    main()
