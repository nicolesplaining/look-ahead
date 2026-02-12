"""Train poem rhyme probes on pre-extracted activation datasets (Step 2).

No language model required; reads .pt files produced by build_dataset.sh.
Writes experiment_results.json compatible with look_ahead_probe.visualize_results.
"""

import argparse
import json
from pathlib import Path

import torch

from .train_poem_probe import train_poem_probes


def main():
    parser = argparse.ArgumentParser(
        description="Train poem rhyme probes on pre-extracted activation datasets"
    )
    parser.add_argument("--train_dataset", type=str, required=True,
                        help="Path to poem activations_train.pt")
    parser.add_argument("--val_dataset", type=str, default=None,
                        help="Path to poem activations_val.pt (optional)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for probes and results")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (default: all)")
    parser.add_argument("--probe_type", type=str, default="linear",
                        choices=["linear", "mlp"])
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probes_dir = output_dir / "probes"

    # Load metadata (no model needed)
    data = torch.load(args.train_dataset, weights_only=False)
    metadata = data['metadata']

    print("\n" + "=" * 80)
    print("POEM PROBE TRAINING (decoupled step 2)")
    print("=" * 80)
    print(f"Train dataset: {args.train_dataset}")
    print(f"Val dataset:   {args.val_dataset or '(none)'}")
    print(f"Probe type:    {args.probe_type}")
    print(f"Output dir:    {args.output_dir}")
    print("=" * 80)

    train_poem_probes(
        train_dataset_path=args.train_dataset,
        val_dataset_path=args.val_dataset,
        layers=args.layers,
        probe_type=args.probe_type,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=str(probes_dir),
        device=args.device,
    )

    # Load summary and write experiment_results.json
    summary_path = probes_dir / "training_summary.pt"
    if not summary_path.exists():
        print(f"WARNING: Summary not found at {summary_path}")
        return

    summary = torch.load(summary_path, weights_only=False)
    all_results = {}
    for key, layer_data in summary['all_results'].items():
        layer_idx = layer_data['layer']
        entry = {
            'layer': layer_idx,
            # k=1 is a convention for the poem task so that visualize_results.py
            # (which groups by k) can plot these results without modification.
            'k': 1,
            'train_accuracy': float(layer_data['history']['train_acc'][-1]),
            'train_loss': float(layer_data['history']['train_loss'][-1]),
            'probe_path': layer_data['save_path'],
        }
        if layer_data.get('results') is not None:
            entry['val_accuracy'] = float(layer_data['results']['accuracy'])
            entry['val_top5_accuracy'] = float(layer_data['results']['top5_accuracy'])
            entry['val_loss'] = float(layer_data['results']['loss'])
        all_results[f"layer{layer_idx}_k1"] = entry

    results_data = {
        'config': {
            'model_name': metadata.get('model_name', 'unknown'),
            'task': 'poem_rhyme_prediction',
            'train_dataset': args.train_dataset,
            'val_dataset': args.val_dataset,
            'probe_type': args.probe_type,
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
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
