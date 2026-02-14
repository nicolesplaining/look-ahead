#!/usr/bin/env python3
"""
Visualizes experimental results from layer-k probe experiments.
Accepts one or more result JSONs and overlays them on a single plot per k value.
"""

import json
import argparse
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_results(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def organize_results_by_k(results):
    """Returns {k: [(layer, metrics_dict), ...]} sorted by layer."""
    results_by_k = defaultdict(list)
    for key, value in results.items():
        if key.startswith('layer'):
            layer = value['layer']
            k = value.get('k')
            metrics = {
                'train_accuracy':    value.get('train_accuracy'),
                'val_accuracy':      value.get('val_accuracy'),
                'val_top5_accuracy': value.get('val_top5_accuracy'),
            }
            results_by_k[k].append((layer, metrics))
    for k in results_by_k:
        results_by_k[k].sort(key=lambda x: x[0])
    return results_by_k


def plot_results(all_results_by_k, labels, colors, output_dir,
                 show_val=False, show_train=False, show_top5=False,
                 acc_min=0.0, acc_max=1.0):
    """
    Overlay multiple result sets on a single plot per k value.

    Color  → dataset (one per JSON)
    Style  → metric type (solid=val, dashed=train, dotted=top-5)
    """
    # Resolve colors: fill unspecified slots from matplotlib's default cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    default_colors = [p['color'] for p in prop_cycle]
    resolved_colors = [
        c if c else default_colors[i % len(default_colors)]
        for i, c in enumerate(colors)
    ]

    # Which metrics to plot and their visual style
    metric_specs = []
    if show_val:   metric_specs.append(('val_accuracy',      'Val',   'o', '-'))
    if show_train: metric_specs.append(('train_accuracy',    'Train', 's', '--'))
    if show_top5:  metric_specs.append(('val_top5_accuracy', 'Top-5', '^', ':'))

    multi_metric = len(metric_specs) > 1

    all_k = sorted(set(
        k for r in all_results_by_k for k in r.keys() if k is not None
    ))

    filename_parts = [name.lower().replace('-', '') for _, name, _, _ in metric_specs]

    for k in all_k:
        fig, ax = plt.subplots(figsize=(10, 6))
        plotted = False

        for results_by_k, label, color in zip(all_results_by_k, labels, resolved_colors):
            if k not in results_by_k:
                continue

            layers = [layer for layer, _ in results_by_k[k]]

            for metric_key, metric_name, marker, linestyle in metric_specs:
                vals = [
                    m[metric_key]
                    for _, m in results_by_k[k]
                    if m.get(metric_key) is not None
                ]
                if not vals:
                    continue
                legend_label = f"{label} ({metric_name})" if multi_metric else label
                ax.plot(layers, vals, linewidth=2,
                        linestyle=linestyle, color=color, label=legend_label)
                plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f"Accuracy Across Layers (k={k})", fontsize=14, fontweight='bold')
        ax.set_ylim(acc_min, acc_max)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        fig.tight_layout()

        filename = f"{'_'.join(filename_parts)}_accuracy_k{k}.png"
        output_path = Path(output_dir) / filename
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Overlay multiple layer-k probe result JSONs on a single plot per k.'
    )
    parser.add_argument(
        'results_json',
        nargs='+',
        help='One or more paths to experiment result JSON files'
    )
    parser.add_argument(
        '--labels',
        nargs='+',
        default=None,
        help='Legend label for each JSON (default: filename stem). Must match number of JSONs if provided.'
    )
    parser.add_argument(
        '--colors',
        nargs='+',
        default=None,
        help='Matplotlib color for each JSON (default: auto). Must match number of JSONs if provided.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save plots (default: directory of the first JSON)'
    )
    parser.add_argument('--show-val',   action='store_true', help='Plot validation accuracy')
    parser.add_argument('--show-train', action='store_true', help='Plot training accuracy')
    parser.add_argument('--show-top5',  action='store_true', help='Plot top-5 validation accuracy')
    parser.add_argument('--acc-min', type=float, default=0.0, help='Y-axis lower bound (default: 0.0)')
    parser.add_argument('--acc-max', type=float, default=1.0, help='Y-axis upper bound (default: 1.0)')

    args = parser.parse_args()

    if not (args.show_val or args.show_train or args.show_top5):
        print("ERROR: specify at least one of --show-val, --show-train, --show-top5")
        sys.exit(1)

    n = len(args.results_json)

    labels = args.labels or [Path(p).stem for p in args.results_json]
    if len(labels) != n:
        print(f"ERROR: --labels count ({len(labels)}) must match number of JSONs ({n})")
        sys.exit(1)

    colors = list(args.colors) if args.colors else [None] * n
    if len(colors) != n:
        print(f"ERROR: --colors count ({len(colors)}) must match number of JSONs ({n})")
        sys.exit(1)

    output_dir = args.output_dir or str(Path(args.results_json[0]).parent)

    all_results_by_k = []
    for path in args.results_json:
        print(f"Loading: {path}")
        data = load_results(path)
        all_results_by_k.append(organize_results_by_k(data['results']))

    all_k = sorted(set(
        k for r in all_results_by_k for k in r.keys() if k is not None
    ))
    print(f"k values: {all_k}")
    print(f"Output dir: {output_dir}\n")

    plot_results(all_results_by_k, labels, colors, output_dir,
                 show_val=args.show_val,
                 show_train=args.show_train,
                 show_top5=args.show_top5,
                 acc_min=args.acc_min,
                 acc_max=args.acc_max)

    print("Done!")


if __name__ == '__main__':
    main()
