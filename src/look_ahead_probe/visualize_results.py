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


def organize_results_by_k(results, include_train=False):
    """
    Organize results by k value.

    Returns:
        dict: {k_value: [(layer, val_accuracy, train_accuracy), ...]} if include_train
              {k_value: [(layer, val_accuracy), ...]} otherwise
    """
    results_by_k = defaultdict(list)

    for key, value in results.items():
        if key.startswith('layer'):
            layer = value['layer']
            k = value['k']
            val_accuracy = value['val_accuracy']

            if include_train:
                train_accuracy = value['train_accuracy']
                results_by_k[k].append((layer, val_accuracy, train_accuracy))
            else:
                results_by_k[k].append((layer, val_accuracy))

    # Sort by layer for each k
    for k in results_by_k:
        results_by_k[k].sort(key=lambda x: x[0])

    return results_by_k


def plot_results(results_by_k, output_dir, show_train=False):
    """
    Create separate plots for each k value.

    Args:
        results_by_k: Dictionary mapping k values to (layer, val_accuracy[, train_accuracy]) tuples
        output_dir: Directory to save plots
        show_train: Whether to include training accuracy on the plot
    """
    k_values = sorted(results_by_k.keys())

    for k in k_values:
        plt.figure(figsize=(10, 6))

        if show_train:
            layers, val_accuracies, train_accuracies = zip(*results_by_k[k])
            plt.plot(layers, val_accuracies, marker='o', linewidth=2, markersize=6, label='Validation')
            plt.plot(layers, train_accuracies, marker='s', linewidth=2, markersize=6, label='Train')
            plt.legend(fontsize=11)
            title = f'Train & Validation Accuracy Across Layers (k={k})'
            filename = f'train_val_accuracy_k{k}.png'
        else:
            layers, val_accuracies = zip(*results_by_k[k])
            plt.plot(layers, val_accuracies, marker='o', linewidth=2, markersize=6)
            title = f'Validation Accuracy Across Layers (k={k})'
            filename = f'val_accuracy_k{k}.png'

        plt.xlabel('Layer', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
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
        '--show-train',
        action='store_true',
        help='Include training accuracy on plots alongside validation accuracy'
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results_json}")
    data = load_results(args.results_json)

    # Organize by k
    results_by_k = organize_results_by_k(data['results'], include_train=args.show_train)
    print(f"Found results for k values: {sorted(results_by_k.keys())}")

    # Determine output directory
    output_dir = args.output_dir
    if output_dir is None:
        # Save to same dir as JSON by default
        output_dir = Path(args.results_json).parent

    # Create plots
    plot_results(results_by_k, output_dir, show_train=args.show_train)

    print("Done!")


if __name__ == '__main__':
    main()
