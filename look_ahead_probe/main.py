"""Train look-ahead probes from pre-extracted activations."""

import argparse
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from .data_loading import ActivationDataset, load_extracted_dataset
from .probe import FutureTokenProbe
from .train_probe import train_probe, evaluate_probe


def train_from_extracted_dataset(
    dataset_path: str,
    layer_idx: int,
    k: int,
    probe_type: str = "linear",
    num_epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    val_dataset_path: Optional[str] = None,
    save_path: Optional[str] = None,
    device: str = "cuda"
):
    """Train probe from extracted dataset."""
    print(f"Loading training dataset from: {dataset_path}")
    train_activations, train_targets, metadata = load_extracted_dataset(
        dataset_path, layer_idx=layer_idx, k=k
    )

    print(f"\nTraining data:")
    print(f"  Layer {layer_idx}, k={k}")
    print(f"  Activations: {train_activations.shape}")
    print(f"  Targets: {train_targets.shape}")

    train_dataset = ActivationDataset(train_activations, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_dataset_path is not None:
        val_activations, val_targets, _ = load_extracted_dataset(
            val_dataset_path, layer_idx=layer_idx, k=k
        )
        print(f"Validation data: {val_activations.shape}")

        val_dataset = ActivationDataset(val_activations, val_targets)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    probe = FutureTokenProbe(
        input_dim=metadata['d_model'],
        vocab_size=metadata['vocab_size'],
        probe_type=probe_type
    )

    print(f"\nTraining {probe_type} probe...")
    history = train_probe(
        probe=probe,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device
    )

    results = None
    if val_loader is not None:
        print("\nEvaluating...")
        results = evaluate_probe(probe, val_loader, device)
        print(f"Results: Loss={results['loss']:.4f}, Acc={results['accuracy']:.4f}, Top-5={results['top5_accuracy']:.4f}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'probe_state_dict': probe.state_dict(),
            'layer_idx': layer_idx,
            'k': k,
            'probe_type': probe_type,
            'metadata': metadata,
            'history': history,
            'results': results,
        }, save_path)
        print(f"Saved to: {save_path}")

    return probe, history, results


def main():
    parser = argparse.ArgumentParser(description="Train probe from extracted dataset")

    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--val_dataset", type=str, default=None)

    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--k", type=int, required=True, help="Lookahead distance")
    parser.add_argument("--probe_type", type=str, default="linear", choices=["linear", "mlp"])

    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    train_from_extracted_dataset(
        dataset_path=args.train_dataset,
        layer_idx=args.layer,
        k=args.k,
        probe_type=args.probe_type,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_dataset_path=args.val_dataset,
        save_path=args.save_path,
        device=args.device
    )

    print("Done!")


if __name__ == "__main__":
    main()
