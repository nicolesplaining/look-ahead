"""Batch train probes on all layers."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data_loading import ActivationDataset, load_extracted_dataset
from .probe import FutureTokenProbe
from .train_probe import train_probe, evaluate_probe


def train_all_layers(
    train_dataset_path: str,
    k: int,
    val_dataset_path: str = None,
    layers: list = None,
    probe_type: str = "linear",
    num_epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    output_dir: str = "./trained_probes",
    device: str = "cuda"
):
    """Train probes on all specified layers."""
    print("Loading datasets...")
    train_layer_acts, train_targets, metadata = load_extracted_dataset(
        train_dataset_path, layer_idx=None, k=k
    )

    available_layers = sorted(train_layer_acts.keys())
    if layers is None:
        layers = available_layers
    else:
        for layer in layers:
            if layer not in available_layers:
                raise ValueError(f"Layer {layer} not in dataset. Available: {available_layers}")

    print(f"Training on layers: {layers}, k={k}")

    val_layer_acts = None
    val_targets = None
    if val_dataset_path is not None:
        val_layer_acts, val_targets, _ = load_extracted_dataset(
            val_dataset_path, layer_idx=None, k=k
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for layer_idx in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}")
        print(f"{'='*60}")

        train_acts = train_layer_acts[layer_idx]
        train_dataset = ActivationDataset(train_acts, train_targets)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if val_layer_acts is not None:
            val_acts = val_layer_acts[layer_idx]
            val_dataset = ActivationDataset(val_acts, val_targets)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        probe = FutureTokenProbe(
            input_dim=metadata['d_model'],
            vocab_size=metadata['vocab_size'],
            probe_type=probe_type
        )

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
            results = evaluate_probe(probe, val_loader, device)
            print(f"Results: Loss={results['loss']:.4f}, Acc={results['accuracy']:.4f}, Top-5={results['top5_accuracy']:.4f}")

        save_path = output_path / f"probe_layer_{layer_idx}_k{k}_{probe_type}.pt"
        torch.save({
            'probe_state_dict': probe.state_dict(),
            'layer_idx': layer_idx,
            'k': k,
            'probe_type': probe_type,
            'metadata': metadata,
            'history': history,
            'results': results,
        }, save_path)

        all_results[layer_idx] = {
            'history': history,
            'results': results,
            'save_path': str(save_path)
        }

    summary_path = output_path / f"training_summary_k{k}.pt"
    torch.save({
        'layers': layers,
        'k': k,
        'probe_type': probe_type,
        'metadata': metadata,
        'all_results': all_results,
    }, summary_path)

    print(f"\n{'='*60}")
    print("Results Comparison:")
    print(f"{'Layer':<10} {'Val Loss':<12} {'Accuracy':<12} {'Top-5 Acc':<12}")
    print("-" * 50)
    for layer_idx in layers:
        res = all_results[layer_idx]['results']
        if res:
            print(f"{layer_idx:<10} {res['loss']:<12.4f} {res['accuracy']:<12.4f} {res['top5_accuracy']:<12.4f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Train probes on all layers")

    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--val_dataset", type=str, default=None)
    parser.add_argument("--k", type=int, required=True, help="Lookahead distance")

    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (default: all)")
    parser.add_argument("--probe_type", type=str, default="linear", choices=["linear", "mlp"])

    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    parser.add_argument("--output_dir", type=str, default="./trained_probes")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    layers = None
    if args.layers is not None:
        layers = [int(x.strip()) for x in args.layers.split(',')]

    train_all_layers(
        train_dataset_path=args.train_dataset,
        k=args.k,
        val_dataset_path=args.val_dataset,
        layers=layers,
        probe_type=args.probe_type,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()
