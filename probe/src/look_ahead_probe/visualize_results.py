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
                 acc_min=0.0, acc_max=1.0, k_values=None):
    """
    One subplot per k value, all side by side in a single figure.

    Color → series (one per JSON)
    k_values: list of ints to include; None means all available k values.
    A dashed vertical separator is drawn between any two subplots whose k
    values are non-consecutive (gap > 1).
    """
    # Resolve colors: fill unspecified slots from matplotlib's default cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    default_colors = [p['color'] for p in prop_cycle]
    resolved_colors = [
        c if c else default_colors[i % len(default_colors)]
        for i, c in enumerate(colors)
    ]

    metric_specs = []
    if show_val:   metric_specs.append(('val_accuracy',      'Val',   '-'))
    if show_train: metric_specs.append(('train_accuracy',    'Train', '--'))
    if show_top5:  metric_specs.append(('val_top5_accuracy', 'Top-5', '-'))

    multi_metric = len(metric_specs) > 1

    available_k = sorted(set(
        k for r in all_results_by_k for k in r.keys() if k is not None
    ))

    if k_values is not None:
        missing = [k for k in k_values if k not in available_k]
        if missing:
            print(f"WARNING: requested k values not found in data: {missing}")
        all_k = [k for k in sorted(k_values) if k in available_k]
    else:
        all_k = available_k

    if not all_k:
        print("No k values found; nothing to plot.")
        return

    filename_parts = [name.lower().replace('-', '') for _, name, _ in metric_specs]

    fig, axes = plt.subplots(1, len(all_k), figsize=(12 * len(all_k), 10), sharey=True)
    if len(all_k) == 1:
        axes = [axes]

    gap_after = [i for i in range(len(all_k) - 1) if all_k[i + 1] - all_k[i] > 1]

    for ax, k in zip(axes, all_k):
        for results_by_k, label, color in zip(all_results_by_k, labels, resolved_colors):
            if k not in results_by_k:
                continue

            layers = [layer for layer, _ in results_by_k[k]]

            for metric_key, metric_name, linestyle in metric_specs:
                vals = [
                    m[metric_key]
                    for _, m in results_by_k[k]
                    if m.get(metric_key) is not None
                ]
                if not vals:
                    continue
                legend_label = f"{label} ({metric_name})" if multi_metric else label
                ax.plot(layers, vals, linewidth=4,
                        linestyle=linestyle, color=color, label=legend_label)

        ax.set_title(f"k={k}", fontsize=40, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=24)
        ax.set_ylim(acc_min, acc_max)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=24)
        ax.legend(fontsize=33, loc='upper left')

    axes[0].set_ylabel('Accuracy', fontsize=24)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Draw a dashed vertical separator between non-consecutive k values
    for i in gap_after:
        bbox_l = axes[i].get_position()
        bbox_r = axes[i + 1].get_position()
        x_sep = (bbox_l.x1 + bbox_r.x0) / 2
        fig.add_artist(plt.Line2D(
            [x_sep, x_sep], [0.03, 0.97],
            transform=fig.transFigure,
            color='gray', linestyle='--', linewidth=2, alpha=0.7,
            clip_on=False,
        ))

    filename = f"{'_'.join(filename_parts)}_accuracy.png"
    output_path = Path(output_dir) / filename
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


def get_unigram_accuracy(unigram_json_path, metric_key='val_accuracy'):
    """Returns the unigram accuracy for the given metric key (single value, same for all k)."""
    with open(unigram_json_path, 'r') as f:
        data = json.load(f)
    results = data.get('results', {})
    for entry in results.values():
        acc = entry.get(metric_key)
        if acc is not None:
            return acc
    return None


def plot_single(results_by_k, output_dir, file_name,
                show_val=False, show_train=False, show_top5=False,
                acc_min=0.0, acc_max=1.0, k_values=None,
                unigram_acc=None, colors=None):
    """
    Single plot: one line per k value, all on one axes.
    X = layer, Y = accuracy.
    If unigram_by_k is provided, draws a horizontal dashed line per k
    at the unigram accuracy for that k (same color as the k line).
    """
    prop_cycle = plt.rcParams['axes.prop_cycle']
    default_colors = [p['color'] for p in prop_cycle]

    available_k = sorted(k for k in results_by_k if k is not None)

    if k_values is not None:
        missing = [k for k in k_values if k not in available_k]
        if missing:
            print(f"WARNING: requested k values not found in data: {missing}")
        all_k = [k for k in sorted(k_values) if k in available_k]
    else:
        all_k = available_k

    if not all_k:
        print("No k values found; nothing to plot.")
        return

    # Resolve per-k colors
    if colors and len(colors) >= len(all_k):
        k_colors = list(colors[:len(all_k)])
    else:
        k_colors = [default_colors[i % len(default_colors)] for i in range(len(all_k))]

    metric_specs = []
    if show_val:   metric_specs.append(('val_accuracy',      'Val',   '-'))
    if show_train: metric_specs.append(('train_accuracy',    'Train', '--'))
    if show_top5:  metric_specs.append(('val_top5_accuracy', 'Top-5', '-'))

    if not metric_specs:
        print("No metric selected; nothing to plot.")
        return

    multi_metric = len(metric_specs) > 1

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    for k, color in zip(all_k, k_colors):
        if k not in results_by_k:
            continue
        layers = [layer for layer, _ in results_by_k[k]]
        for metric_key, metric_name, linestyle in metric_specs:
            vals = [
                m[metric_key]
                for _, m in results_by_k[k]
                if m.get(metric_key) is not None
            ]
            if not vals:
                continue
            label = f"k={k}" + (f" ({metric_name})" if multi_metric else "")
            ax.plot(layers, vals, linewidth=2, linestyle=linestyle,
                    color=color, label=label)

    if unigram_acc is not None:
        ax.axhline(unigram_acc, linestyle='--', linewidth=2,
                   color='gray', alpha=0.7, label='Unigram')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim(acc_min, acc_max)
    ax.grid(True, alpha=0.3)

    # 2-column legend: find the gap between non-consecutive k values and split there.
    # matplotlib fills column-major with ncol=2, so just pass in natural order.
    handles, leg_labels = ax.get_legend_handles_labels()
    k_indices = [i for i, lbl in enumerate(leg_labels) if lbl.startswith('k=')]
    k_vals_in_legend = [int(leg_labels[i].split('=')[1].split()[0]) for i in k_indices]
    has_gap = any(
        k_vals_in_legend[i + 1] - k_vals_in_legend[i] > 1
        for i in range(len(k_indices) - 1)
    )
    if has_gap:
        ax.legend(handles, leg_labels, fontsize=14, loc='upper left', ncol=2)
    else:
        ax.legend(handles, leg_labels, fontsize=11, loc='upper left')
    fig.tight_layout()

    output_path = Path(output_dir) / file_name
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
    parser.add_argument(
        '--k-values',
        nargs='+',
        type=int,
        default=None,
        help='Which k values to include (default: all). E.g. --k-values 1 2 3 8',
    )
    parser.add_argument(
        '--single-plot',
        action='store_true',
        help='Single-plot mode: one line per k on one axes (requires exactly one results JSON). '
             'Use --unigram-json to add a horizontal baseline per k.',
    )
    parser.add_argument(
        '--unigram-json',
        type=str,
        default=None,
        help='Path to unigram baseline JSON (used with --single-plot).',
    )
    parser.add_argument(
        '--file-name',
        type=str,
        default=None,
        help='Output filename (used with --single-plot; default: val_accuracy.png etc.).',
    )

    args = parser.parse_args()

    if not (args.show_val or args.show_train or args.show_top5):
        print("ERROR: specify at least one of --show-val, --show-train, --show-top5")
        sys.exit(1)

    output_dir = args.output_dir or str(Path(args.results_json[0]).parent)
    print(f"Output dir: {output_dir}\n")

    if args.single_plot:
        if len(args.results_json) != 1:
            print("ERROR: --single-plot requires exactly one results JSON")
            sys.exit(1)
        print(f"Loading: {args.results_json[0]}")
        data = load_results(args.results_json[0])
        results_by_k = organize_results_by_k(data['results'])

        unigram_acc = None
        if args.unigram_json:
            print(f"Loading unigram baseline: {args.unigram_json}")
            unigram_metric = ('val_top5_accuracy' if args.show_top5 else
                              'train_accuracy'    if args.show_train else
                              'val_accuracy')
            unigram_acc = get_unigram_accuracy(args.unigram_json, metric_key=unigram_metric)

        metric_tag = ('top5' if args.show_top5 else
                      'train' if args.show_train else 'val')
        file_name = args.file_name or f"{metric_tag}_accuracy.png"

        plot_single(results_by_k, output_dir, file_name,
                    show_val=args.show_val,
                    show_train=args.show_train,
                    show_top5=args.show_top5,
                    acc_min=args.acc_min,
                    acc_max=args.acc_max,
                    k_values=args.k_values,
                    unigram_acc=unigram_acc,
                    colors=list(args.colors) if args.colors else None)
    else:
        n = len(args.results_json)

        labels = args.labels or [Path(p).stem for p in args.results_json]
        if len(labels) != n:
            print(f"ERROR: --labels count ({len(labels)}) must match number of JSONs ({n})")
            sys.exit(1)

        colors = list(args.colors) if args.colors else [None] * n
        if len(colors) != n:
            print(f"ERROR: --colors count ({len(colors)}) must match number of JSONs ({n})")
            sys.exit(1)

        all_results_by_k = []
        for path in args.results_json:
            print(f"Loading: {path}")
            data = load_results(path)
            all_results_by_k.append(organize_results_by_k(data['results']))

        plot_results(all_results_by_k, labels, colors, output_dir,
                     show_val=args.show_val,
                     show_train=args.show_train,
                     show_top5=args.show_top5,
                     acc_min=args.acc_min,
                     acc_max=args.acc_max,
                     k_values=args.k_values)

    print("Done!")


if __name__ == '__main__':
    main()
