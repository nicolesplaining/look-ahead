#!/usr/bin/env python3
"""
Visualizes experimental results from layer-k probe experiments.
Creates separate plots for each k value showing validation accuracy across layers.
"""

import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_results(json_path):
    """Load experiment results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def organize_results_by_k(results):
    """
    Organize results by k value.

    Returns:
        dict: {k_value: [(layer, metrics_dict), ...]}
        where metrics_dict contains available metrics for that layer/k
    """
    results_by_k = defaultdict(list)

    for key, value in results.items():
        if key.startswith('layer'):
            layer = value['layer']
            k = value.get('k')

            # Collect all available metrics
            metrics = {
                'train_accuracy': value.get('train_accuracy'),
                'val_accuracy': value.get('val_accuracy'),
                'val_top5_accuracy': value.get('val_top5_accuracy'),
            }

            results_by_k[k].append((layer, metrics))

    # Sort by layer for each k
    for k in results_by_k:
        results_by_k[k].sort(key=lambda x: x[0])

    return results_by_k


def plot_results(results_by_k, output_dir, show_val=False, show_train=False, show_top5=False):
    """
    Create separate plots for each k value.

    Args:
        results_by_k: Dictionary mapping k values to (layer, metrics_dict) tuples
        output_dir: Directory to save plots
        show_val: Whether to include validation accuracy
        show_train: Whether to include training accuracy
        show_top5: Whether to include top-5 validation accuracy
    """
    k_values = sorted([k for k in results_by_k.keys() if k is not None])

    for k in k_values:
        plt.figure(figsize=(10, 6))

        # Extract layers and metrics
        layers = [layer for layer, _ in results_by_k[k]]

        # Collect which metrics to plot
        plot_count = 0
        title_parts = []
        filename_parts = []

        # Plot validation accuracy if requested
        if show_val:
            val_accs = [metrics['val_accuracy'] for _, metrics in results_by_k[k] if metrics['val_accuracy'] is not None]
            if val_accs:
                plt.plot(layers, val_accs, marker='o', linewidth=2, markersize=6, label='Val Accuracy')
                plot_count += 1
                title_parts.append('Val')
                filename_parts.append('val')

        # Plot training accuracy if requested
        if show_train:
            train_accs = [metrics['train_accuracy'] for _, metrics in results_by_k[k] if metrics['train_accuracy'] is not None]
            if train_accs:
                plt.plot(layers, train_accs, marker='s', linewidth=2, markersize=6, label='Train Accuracy')
                plot_count += 1
                title_parts.append('Train')
                filename_parts.append('train')

        # Plot top-5 accuracy if requested
        if show_top5:
            top5_accs = [metrics['val_top5_accuracy'] for _, metrics in results_by_k[k] if metrics['val_top5_accuracy'] is not None]
            if top5_accs:
                plt.plot(layers, top5_accs, marker='^', linewidth=2, markersize=6, label='Val Top-5 Accuracy')
                plot_count += 1
                title_parts.append('Top-5')
                filename_parts.append('top5')

        # Skip if nothing to plot
        if plot_count == 0:
            plt.close()
            continue

        title = f"{' & '.join(title_parts)} Accuracy Across Layers (k={k})"
        filename = f"{'_'.join(filename_parts)}_accuracy_k{k}.png"

        plt.xlabel('Layer', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)

        if plot_count > 1:
            plt.legend(fontsize=11)

        plt.tight_layout()

        output_path = Path(output_dir) / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize layer-k probe experiment results'
    )
    parser.add_argument(
        'results_json',
        type=str,
        help='Path to experiment results JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save plots (default: same directory as results JSON)'
    )
    parser.add_argument(
        '--show-val',
        action='store_true',
        help='Include validation accuracy on plots'
    )
    parser.add_argument(
        '--show-train',
        action='store_true',
        help='Include training accuracy on plots'
    )
    parser.add_argument(
        '--show-top5',
        action='store_true',
        help='Include top-5 validation accuracy on plots'
    )

    args = parser.parse_args()

    # Check that at least one metric is selected
    if not (args.show_val or args.show_train or args.show_top5):
        print("ERROR: No metrics selected!")
        print("Use at least one of: --show-val, --show-train, --show-top5")
        sys.exit(1)

    # Load results
    print(f"Loading results from: {args.results_json}")
    data = load_results(args.results_json)

    # Organize by k
    results_by_k = organize_results_by_k(data['results'])
    print(f"Found results for k values: {sorted([k for k in results_by_k.keys() if k is not None])}")

    # Determine output directory
    output_dir = args.output_dir
    if output_dir is None:
        # Save to same dir as JSON by default
        output_dir = Path(args.results_json).parent

    # Create plots
    print(f"\nPlotting:")
    if args.show_val:
        print(f"  - Validation accuracy: ✓")
    if args.show_train:
        print(f"  - Training accuracy: ✓")
    if args.show_top5:
        print(f"  - Top-5 accuracy: ✓")
    print()

    plot_results(results_by_k, output_dir,
                 show_val=args.show_val,
                 show_train=args.show_train,
                 show_top5=args.show_top5)

    print("Done!")


if __name__ == '__main__':
    main()
