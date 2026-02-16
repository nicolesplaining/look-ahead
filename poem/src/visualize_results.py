#!/usr/bin/env python3
"""
Visualize probe results for a single experiment_results.json.

Plots one line per selected metric across layers.

Usage:
    python -m visualize_results path/to/experiment_results.json --show-val --show-rhyme5
"""

import json
import argparse
import sys
import matplotlib.pyplot as plt
from pathlib import Path


METRIC_SPECS = {
    'val':    ('val_accuracy',         'Top-1 Val', '-'),
    'top5':   ('val_top5_accuracy',    'Top-5 Val', '--'),
    'rhyme':  ('rhyme_accuracy',       'Rhyme@1',   ':'),
    'rhyme5': ('top5_rhyme_accuracy',  'Rhyme@5',   '-.'),
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
        description='Plot accuracy curves for a single experiment result JSON.'
    )
    parser.add_argument('result_json', help='Path to experiment_results.json')
    parser.add_argument('--show-val',    action='store_true', help='Plot val top-1 accuracy')
    parser.add_argument('--show-top5',   action='store_true', help='Plot val top-5 accuracy')
    parser.add_argument('--show-rhyme',  action='store_true', help='Plot rhyme@1 accuracy')
    parser.add_argument('--show-rhyme5', action='store_true', help='Plot rhyme@5 accuracy')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same dir as JSON)')
    parser.add_argument('--file_name', type=str, default=None,
                        help='Output PNG filename (default: auto-generated from selected metrics)')
    parser.add_argument('--title', type=str, default=None,
                        help='Plot title (default: parent directory name)')
    parser.add_argument('--acc-min', type=float, default=0.0, help='Y-axis lower bound')
    parser.add_argument('--acc-max', type=float, default=1.0, help='Y-axis upper bound')
    args = parser.parse_args()

    selected = []
    if args.show_val:    selected.append('val')
    if args.show_top5:   selected.append('top5')
    if args.show_rhyme:  selected.append('rhyme')
    if args.show_rhyme5: selected.append('rhyme5')

    if not selected:
        print("ERROR: specify at least one of --show-val, --show-top5, --show-rhyme, --show-rhyme5")
        sys.exit(1)

    print(f"Loading: {args.result_json}")
    entries = load_layer_metrics(args.result_json)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = [p['color'] for p in prop_cycle]

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, metric_key in enumerate(selected):
        key, label, linestyle = METRIC_SPECS[metric_key]
        pairs = [(layer, m[key]) for layer, m in entries if m.get(key) is not None]
        if not pairs:
            print(f"  (no data for {metric_key}, skipping)")
            continue
        layers, vals = zip(*pairs)
        ax.plot(layers, vals, linewidth=2, linestyle=linestyle,
                color=colors[idx % len(colors)], label=label)

    title = args.title or Path(args.result_json).parent.name
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(args.acc_min, args.acc_max)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()

    output_dir = args.output_dir or str(Path(args.result_json).parent)
    if args.file_name:
        filename = args.file_name if args.file_name.endswith('.png') else args.file_name + '.png'
    else:
        filename = '_'.join(selected) + '.png'
    output_path = Path(output_dir) / filename
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
