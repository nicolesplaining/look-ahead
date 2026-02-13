"""Train probes on pre-extracted activation datasets (Step 2 of decoupled pipeline).

No language model is required; reads .pt files produced by build_dataset.sh.
Writes experiment_results.json in the same schema as layer_k_experiment.py.
"""

import argparse
import json
from pathlib import Path

import torch

from .train_all_layers import train_all_layers


def main():
    parser = argparse.ArgumentParser(
        description="Train probes on pre-extracted activation datasets"
    )
    parser.add_argument("--train_dataset", type=str, required=True,
                        help="Path to activations_train.pt")
    parser.add_argument("--val_dataset", type=str, default=None,
                        help="Path to activations_val.pt (optional)")
    parser.add_argument("--max_k", type=int, required=True,
                        help="Maximum lookahead distance")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for probes and results")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (default: all)")
    parser.add_argument("--probe_type", type=str, default="linear",
                        choices=["linear", "mlp"])
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--save_weights", action="store_true", default=False,
                        help="Save probe weight files (large: ~3GB each for 32B models)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    layers = None
    if args.layers is not None:
        layers = [int(x.strip()) for x in args.layers.split(',')]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probes_dir = output_dir / "probes"
    probes_dir.mkdir(exist_ok=True)

    # Load metadata from train dataset (lightweight: just read the dict, no model needed)
    data = torch.load(args.train_dataset, weights_only=False)
    metadata = data['metadata']

    print("\n" + "=" * 80)
    print("PROBE TRAINING (decoupled step 2)")
    print("=" * 80)
    print(f"Train dataset: {args.train_dataset}")
    print(f"Val dataset:   {args.val_dataset or '(none)'}")
    print(f"K range:       1 to {args.max_k}")
    print(f"Probe type:    {args.probe_type}")
    print(f"Output dir:    {args.output_dir}")
    print("=" * 80)

    all_results = {}

    for k in range(1, args.max_k + 1):
        print(f"\n{'='*80}")
        print(f"Training probes for k={k}")
        print(f"{'='*80}")

        k_output_dir = probes_dir / f"k{k}"

        train_all_layers(
            train_dataset_path=args.train_dataset,
            k=k,
            val_dataset_path=args.val_dataset,
            layers=layers,
            probe_type=args.probe_type,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            save_weights=args.save_weights,
            output_dir=str(k_output_dir),
            device=args.device,
        )

        summary_path = k_output_dir / f"training_summary_k{k}.pt"
        if not summary_path.exists():
            print(f"WARNING: Summary file not found at {summary_path}")
            continue

        summary = torch.load(summary_path, weights_only=False)
        for layer_idx, layer_data in summary['all_results'].items():
            if layer_data.get('results') is not None:
                all_results[f"layer{layer_idx}_k{k}"] = {
                    'layer': layer_idx,
                    'k': k,
                    'train_accuracy': float(layer_data['history']['train_acc'][-1]),
                    'train_loss': float(layer_data['history']['train_loss'][-1]),
                    'val_accuracy': float(layer_data['results']['accuracy']),
                    'val_top5_accuracy': float(layer_data['results']['top5_accuracy']),
                    'val_loss': float(layer_data['results']['loss']),
                    'probe_path': layer_data['save_path'],
                }
            elif layer_data.get('history') is not None:
                all_results[f"layer{layer_idx}_k{k}"] = {
                    'layer': layer_idx,
                    'k': k,
                    'train_accuracy': float(layer_data['history']['train_acc'][-1]),
                    'train_loss': float(layer_data['history']['train_loss'][-1]),
                    'probe_path': layer_data['save_path'],
                }

    results_data = {
        'config': {
            'model_name': metadata.get('model_name', 'unknown'),
            'train_dataset': args.train_dataset,
            'val_dataset': args.val_dataset,
            'max_k': args.max_k,
            'probe_type': args.probe_type,
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
        },
        'metadata': metadata,
        'results': all_results,
    }

    results_file = output_dir / 'experiment_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total probes trained: {len(all_results)}")
    print(f"Results saved to:     {results_file}")
    print(f"Probes saved in:      {probes_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
