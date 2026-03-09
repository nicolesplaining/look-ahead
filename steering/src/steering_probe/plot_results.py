#!/usr/bin/env python3
"""
Step 3: Plot heatmaps from steering experiment results.

Usage:
    python -m steering_probe.plot_results \
        --results-path <results/results.json> \
        --output-dir <results/plots/> \
        [--scheme-names '0:at,1:ight,2:ing,3:old,4:and']
"""
import argparse
import json


def parse_args():
    p = argparse.ArgumentParser(description="Plot steering heatmaps")
    p.add_argument("--results-path", required=True, help="Path to results.json")
    p.add_argument("--output-dir", required=True, help="Directory for output PNG files")
    p.add_argument(
        "--scheme-names", default=None,
        help="Comma-separated id:name pairs, e.g. '0:at,1:ight,2:ing' "
             "(default: use numeric IDs)",
    )
    p.add_argument(
        "--figsize", default=None,
        help="Figure size as WxH, e.g. '14x5' (default: auto)",
    )
    p.add_argument(
        "--title", default=None,
        help="Plot title (default: auto-generated)",
    )
    p.add_argument("--xlabel", default=None, help="X-axis label")
    p.add_argument("--ylabel", default=None, help="Y-axis label")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.results_path) as f:
        results = json.load(f)

    scheme_names = {}
    if args.scheme_names:
        for pair in args.scheme_names.split(","):
            k, v = pair.strip().split(":")
            scheme_names[int(k)] = v.strip()

    figsize = None
    if args.figsize:
        w, h = args.figsize.lower().split("x")
        figsize = (float(w), float(h))

    from .plot import plot_all_pairs
    plot_all_pairs(results, scheme_names, args.output_dir,
                   figsize=figsize, title=args.title,
                   xlabel=args.xlabel, ylabel=args.ylabel)


if __name__ == "__main__":
    main()
