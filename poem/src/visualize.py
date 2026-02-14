#!/usr/bin/env python3
"""
Visualize poem probe results.

Accepts one or more experiment_results.json files (typically one per i-position)
and overlays them on a single plot per i-value.

Color  → dataset / i-position (one per JSON)
Style  → metric type (solid=val, dashed=top-5, dotted=rhyme)
"""

import json
import argparse
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_results(json_path):
    with open(json_path) as f:
        return json.load(f)


def organize_results_by_i(results):
    """Returns {i_val: [(layer, metrics_dict), ...]} sorted by layer."""
    results_by_i = defaultdict(list)
    for key, value in results.items():
        if key.startswith('layer'):
            layer = value['layer']
            i_val = value.get('k')   # 'k' stores train_position (the i value)
            metrics = {
                'val_accuracy':      value.get('val_accuracy'),
                'val_top5_accuracy': value.get('val_top5_accuracy'),
                'rhyme_accuracy':    value.get('rhyme_accuracy'),
            }
            results_by_i[i_val].append((layer, metrics))
    for i in results_by_i:
        results_by_i[i].sort(key=lambda x: x[0])
    return results_by_i


def plot_results(all_results_by_i, labels, colors, output_dir,
                 show_val=False, show_top5=False, show_rhyme=False,
                 acc_min=0.0, acc_max=1.0):
    # Resolve unspecified colors from matplotlib's default cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    default_colors = [p['color'] for p in prop_cycle]
    resolved_colors = [
        c if c else default_colors[idx % len(default_colors)]
        for idx, c in enumerate(colors)
    ]

    metric_specs = []
    if show_val:   metric_specs.append(('val_accuracy',      'Val',    '-'))
    if show_top5:  metric_specs.append(('val_top5_accuracy', 'Top-5',  '--'))
    if show_rhyme: metric_specs.append(('rhyme_accuracy',    'Rhyme%', ':'))

    multi_metric = len(metric_specs) > 1

    all_i = sorted(set(
        i for r in all_results_by_i for i in r.keys() if i is not None
    ))

    for i_val in all_i:
        fig, ax = plt.subplots(figsize=(10, 6))
        plotted = False

        for results_by_i, label, color in zip(all_results_by_i, labels, resolved_colors):
            if i_val not in results_by_i:
                continue
            layers = [layer for layer, _ in results_by_i[i_val]]
            for metric_key, metric_name, linestyle in metric_specs:
                vals = [
                    m[metric_key]
                    for _, m in results_by_i[i_val]
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
        ax.set_title(f"Accuracy Across Layers (i={i_val})", fontsize=14, fontweight='bold')
        ax.set_ylim(acc_min, acc_max)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        fig.tight_layout()

        i_str = f"i{i_val}" if i_val >= 0 else f"i_neg{abs(i_val)}"
        metric_parts = [name.lower().replace('-', '').replace('%', '')
                        for _, name, _ in metric_specs]
        filename = f"{'_'.join(metric_parts)}_accuracy_{i_str}.png"
        output_path = Path(output_dir) / filename
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Overlay poem probe result JSONs on a single plot per i-position.'
    )
    parser.add_argument(
        'results_json',
        nargs='+',
        help='One or more paths to experiment_results.json files'
    )
    parser.add_argument(
        '--labels',
        nargs='+',
        default=None,
        help='Legend label for each JSON (default: parent directory name, e.g. "i0", "i_neg3"). '
             'Must match number of JSONs if provided.'
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
        help='Directory to save plots (default: parent of the parent of the first JSON)'
    )
    parser.add_argument('--show-val',   action='store_true', help='Plot val accuracy')
    parser.add_argument('--show-top5',  action='store_true', help='Plot top-5 val accuracy')
    parser.add_argument('--show-rhyme', action='store_true', help='Plot rhyme accuracy')
    parser.add_argument('--acc-min', type=float, default=0.0, help='Y-axis lower bound (default: 0.0)')
    parser.add_argument('--acc-max', type=float, default=1.0, help='Y-axis upper bound (default: 1.0)')

    args = parser.parse_args()

    if not (args.show_val or args.show_top5 or args.show_rhyme):
        print("ERROR: specify at least one of --show-val, --show-top5, --show-rhyme")
        sys.exit(1)

    n = len(args.results_json)

    # Default label: parent directory name ("i0", "i_neg3", ...) since all
    # files share the same stem "experiment_results".
    labels = args.labels or [Path(p).parent.name for p in args.results_json]
    if len(labels) != n:
        print(f"ERROR: --labels count ({len(labels)}) must match number of JSONs ({n})")
        sys.exit(1)

    colors = list(args.colors) if args.colors else [None] * n
    if len(colors) != n:
        print(f"ERROR: --colors count ({len(colors)}) must match number of JSONs ({n})")
        sys.exit(1)

    # Default output: one level above the first JSON's parent (the results_linear dir)
    output_dir = args.output_dir or str(Path(args.results_json[0]).parent.parent)

    all_results_by_i = []
    for path in args.results_json:
        print(f"Loading: {path}")
        data = load_results(path)
        all_results_by_i.append(organize_results_by_i(data['results']))

    all_i = sorted(set(
        i for r in all_results_by_i for i in r.keys() if i is not None
    ))
    print(f"i values: {all_i}")
    print(f"Output dir: {output_dir}\n")

    plot_results(all_results_by_i, labels, colors, output_dir,
                 show_val=args.show_val,
                 show_top5=args.show_top5,
                 show_rhyme=args.show_rhyme,
                 acc_min=args.acc_min,
                 acc_max=args.acc_max)

    print("Done!")


if __name__ == '__main__':
    main()
