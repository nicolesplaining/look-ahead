#!/usr/bin/env python3
"""
Compare a single metric across multiple experiment_results.json files.

Plots one line per JSON with configurable color and linestyle.

Usage:
    python -m compare_results i0/experiment_results.json i1/experiment_results.json \
        --metric rhyme5 \
        --labels "i=0" "i=1" \
        --colors tomato steelblue \
        --linestyles "-" "--"
"""

import json
import argparse
import sys
import matplotlib.pyplot as plt
from pathlib import Path


LINESTYLE_ALIASES = {
    'solid':   '-',
    'dashed':  '--',
    'dotted':  ':',
    'dashdot': '-.',
}


def resolve_linestyle(s):
    return LINESTYLE_ALIASES.get(s, s)


METRIC_KEYS = {
    'val':    ('val_accuracy',        'Top-1 Val'),
    'top5':   ('val_top5_accuracy',   'Top-5 Val'),
    'rhyme':  ('rhyme_accuracy',      'Rhyme@1'),
    'rhyme5': ('top5_rhyme_accuracy', 'Rhyme@5'),
}


def load_layer_metrics(json_path):
    with open(json_path) as f:
        data = json.load(f)
    entries = []
    for key, value in data['results'].items():
        if key.startswith('layer'):
            entries.append((value['layer'], value))
    entries.sort(key=lambda x: x[0])
    return entries


def main():
    parser = argparse.ArgumentParser(
        description='Compare one metric across multiple experiment result JSONs.'
    )
    parser.add_argument('result_jsons', nargs='+', help='Paths to experiment_results.json files')
    parser.add_argument('--metric', required=True, choices=['val', 'top5', 'rhyme', 'rhyme5'],
                        help='Metric to plot')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Legend label per JSON (default: parent directory name)')
    parser.add_argument('--colors', nargs='+', default=None,
                        help='Matplotlib color per JSON (default: auto)')
    parser.add_argument('--linestyles', nargs='+', default=None,
                        help='Matplotlib linestyle per JSON (default: "-")')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: parent of first JSON)')
    parser.add_argument('--file_name', type=str, default=None,
                        help='Output PNG filename (default: compare_<metric>.png)')
    parser.add_argument('--title', type=str, default=None,
                        help='Plot title (default: metric label)')
    parser.add_argument('--acc-min', type=float, default=0.0, help='Y-axis lower bound')
    parser.add_argument('--acc-max', type=float, default=1.0, help='Y-axis upper bound')
    args = parser.parse_args()

    n = len(args.result_jsons)
    metric_key, metric_label = METRIC_KEYS[args.metric]

    prop_cycle = plt.rcParams['axes.prop_cycle']
    default_colors = [p['color'] for p in prop_cycle]

    labels     = args.labels     or [Path(p).parent.name for p in args.result_jsons]
    colors     = list(args.colors)     if args.colors     else [default_colors[i % len(default_colors)] for i in range(n)]
    linestyles = [resolve_linestyle(s) for s in args.linestyles] if args.linestyles else ['-'] * n

    for name, lst in [('--labels', labels), ('--colors', colors), ('--linestyles', linestyles)]:
        if len(lst) != n:
            print(f"ERROR: {name} count ({len(lst)}) must match number of JSONs ({n})")
            sys.exit(1)

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False

    for path, label, color, ls in zip(args.result_jsons, labels, colors, linestyles):
        print(f"Loading: {path}")
        entries = load_layer_metrics(path)
        pairs = [(layer, m[metric_key]) for layer, m in entries if m.get(metric_key) is not None]
        if not pairs:
            print(f"  (no data for {args.metric}, skipping)")
            continue
        layers, vals = zip(*pairs)
        ax.plot(layers, vals, linewidth=2, linestyle=ls, color=color, label=label)
        plotted = True

    if not plotted:
        print("ERROR: no data to plot")
        sys.exit(1)

    title = args.title or metric_label
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(args.acc_min, args.acc_max)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()

    output_dir = args.output_dir or str(Path(args.result_jsons[0]).parent)
    if args.file_name:
        filename = args.file_name if args.file_name.endswith('.png') else args.file_name + '.png'
    else:
        filename = f"compare_{args.metric}.png"
    output_path = Path(output_dir) / filename
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
