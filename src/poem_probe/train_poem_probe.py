#!/usr/bin/env python3
"""
Train probes to predict rhyming words from first-line activations.

Reuses probe architectures and training logic from look_ahead_probe.
"""

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import from look_ahead_probe
from look_ahead_probe.probe import FutureTokenProbe
from look_ahead_probe.train_probe import train_probe, evaluate_probe


def load_poem_dataset(dataset_path: str, layer_idx: int):
    """
    Load poem dataset for a specific layer.

    Args:
        dataset_path: Path to .pt dataset file
        layer_idx: Layer to load activations from

    Returns:
        activations: [n_samples, d_model]
        targets: [n_samples]
        metadata: Dict with dataset info
    """
    data = torch.load(dataset_path, weights_only=False)

    layer_activations = data['layer_activations']
    targets = data['targets']
    metadata = data['metadata']

    if layer_idx not in layer_activations:
        raise ValueError(
            f"Layer {layer_idx} not in dataset. "
            f"Available layers: {sorted(layer_activations.keys())}"
        )

    activations = layer_activations[layer_idx]

    print(f"Loaded layer {layer_idx}:")
    print(f"  Activations: {activations.shape}")
    print(f"  Targets: {targets.shape}")
    print(f"  d_model: {metadata['d_model']}")
    print(f"  Vocab size: {metadata['vocab_size']}")

    return activations, targets, metadata


def train_poem_probes(
    train_dataset_path: str,
    val_dataset_path: str = None,
    layers: str = None,
    probe_type: str = "linear",
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    output_dir: str = "poem_probes",
    device: str = "cuda"
):
    """
    Train probes on all layers for poem rhyme prediction.

    Args:
        train_dataset_path: Path to training dataset
        val_dataset_path: Path to validation dataset (optional)
        layers: Comma-separated layer indices or None for all
        probe_type: "linear" or "mlp"
        num_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        output_dir: Where to save probes
        device: Device to use
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata to determine available layers
    data = torch.load(train_dataset_path, weights_only=False)
    metadata = data['metadata']
    available_layers = metadata['layers']

    if layers is not None:
        layers_to_train = [int(x.strip()) for x in layers.split(',')]
    else:
        layers_to_train = available_layers

    print("=" * 80)
    print("POEM RHYME PROBE TRAINING")
    print("=" * 80)
    print(f"Training dataset: {train_dataset_path}")
    print(f"Validation dataset: {val_dataset_path if val_dataset_path else 'None'}")
    print(f"Layers: {layers_to_train}")
    print(f"Probe type: {probe_type}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Output: {output_dir}")
    print("=" * 80 + "\n")

    # Results storage
    all_results = {}

    for layer_idx in layers_to_train:
        print(f"\n{'=' * 80}")
        print(f"TRAINING LAYER {layer_idx}")
        print("=" * 80)

        # Load datasets
        train_acts, train_targets, metadata = load_poem_dataset(
            train_dataset_path, layer_idx
        )

        if val_dataset_path:
            val_acts, val_targets, _ = load_poem_dataset(val_dataset_path, layer_idx)
        else:
            val_acts, val_targets = None, None

        # Create probe
        probe = FutureTokenProbe(
            input_dim=metadata['d_model'],
            vocab_size=metadata['vocab_size'],
            probe_type=probe_type
        ).to(device)

        # Create dataloaders
        train_dataset = TensorDataset(train_acts, train_targets)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        if val_acts is not None:
            val_dataset = TensorDataset(val_acts, val_targets)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        else:
            val_loader = None

        # Train probe (reusing look_ahead_probe training logic)
        print(f"\nTraining probe...")
        history = train_probe(
            probe=probe,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device
        )

        # Evaluate on validation set if available
        results = None
        if val_loader is not None:
            print(f"\nEvaluating on validation set...")
            results = evaluate_probe(probe, val_loader, device=device)
            print(f"  Validation Accuracy: {results['accuracy']:.4f}")
            print(f"  Validation Top-5 Accuracy: {results['top5_accuracy']:.4f}")
            print(f"  Validation Loss: {results['loss']:.4f}")

        # Save probe
        probe_path = output_dir / f"layer{layer_idx}_probe.pt"
        torch.save({
            'probe_state_dict': probe.state_dict(),
            'layer_idx': layer_idx,
            'probe_type': probe_type,
            'metadata': metadata,
        }, probe_path)

        print(f"\n✓ Probe saved to {probe_path}")

        # Store results
        all_results[f"layer{layer_idx}"] = {
            'layer': layer_idx,
            'history': history,
            'results': results,
            'save_path': str(probe_path),
        }

    # Save summary
    summary_path = output_dir / "training_summary.pt"
    torch.save({
        'all_results': all_results,
        'config': {
            'probe_type': probe_type,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
        },
        'metadata': metadata,
    }, summary_path)

    print(f"\n{'=' * 80}")
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Summary saved to: {summary_path}")
    print("\nResults by layer:")
    for layer_idx in layers_to_train:
        key = f"layer{layer_idx}"
        if key in all_results:
            result = all_results[key]
            train_acc = result['history']['train_acc'][-1]
            print(f"  Layer {layer_idx}: Train Acc = {train_acc:.4f}", end="")
            if result['results']:
                val_acc = result['results']['accuracy']
                print(f", Val Acc = {val_acc:.4f}")
            else:
                print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Train poem rhyme prediction probes"
    )

    parser.add_argument("--train_dataset", type=str, required=True,
                        help="Path to training dataset (.pt)")
    parser.add_argument("--val_dataset", type=str, default=None,
                        help="Path to validation dataset (.pt, optional)")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (None = all)")
    parser.add_argument("--probe_type", type=str, default="linear",
                        choices=["linear", "mlp"],
                        help="Probe architecture")
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="poem_probes",
                        help="Output directory")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device")

    args = parser.parse_args()

    train_poem_probes(
        train_dataset_path=args.train_dataset,
        val_dataset_path=args.val_dataset,
        layers=args.layers,
        probe_type=args.probe_type,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()
