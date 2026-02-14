"""Train poem rhyme probes at a specific position i (Step 2).

No language model required; reads .pt files produced by build_dataset.sh.
Writes experiment_results.json compatible with look_ahead_probe.visualize_results.

To compare multiple positions, run this script once per position and overlay the
resulting JSONs in plot_results.sh (visualize_results.py accepts multiple JSONs).
"""

import argparse
import json
from pathlib import Path

import torch

from train_poem_probe import train_all_layers_at_position, make_experiment_results_json


def main():
    parser = argparse.ArgumentParser(
        description="Train poem rhyme probes at a specific position i"
    )
    parser.add_argument("--train_dataset", type=str, required=True,
                        help="Path to poem activations_train.pt")
    parser.add_argument("--val_dataset", type=str, default=None,
                        help="Path to poem activations_val.pt (optional)")
    parser.add_argument("--train_position", type=int, required=True,
                        help="Position i to train on (0=first-line \\n, negative=earlier in first line, "
                             "positive=second line). Must exist in the dataset.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results (a subdirectory i{N} is created inside)")
    parser.add_argument("--probe_type", type=str, default="linear",
                        choices=["linear", "mlp"])
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--save_weights", action="store_true",
                        help="Save probe weights to disk (off by default)")
    parser.add_argument("--model_name", type=str, default=None,
                        help="HuggingFace model name for tokenizer (enables decoded_predictions in JSON)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Each position gets its own subdirectory so runs don't clobber each other
    i_label = f"i{args.train_position}" if args.train_position >= 0 else f"i_neg{abs(args.train_position)}"
    run_dir = Path(args.output_dir) / i_label
    run_dir.mkdir(parents=True, exist_ok=True)
    probes_dir = run_dir / "probes"

    # Optionally load tokenizer for decoding predictions → English
    tokenizer = None
    if args.model_name:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        print("✓ Tokenizer loaded\n")

    # Load metadata (no model needed)
    data = torch.load(args.train_dataset, weights_only=False)
    metadata = data['metadata']
    i_range = metadata.get('i_range', ['?', '?'])

    print("\n" + "=" * 70)
    print("POEM PROBE TRAINING")
    print("=" * 70)
    print(f"Train dataset:  {args.train_dataset}")
    print(f"Val dataset:    {args.val_dataset or '(none)'}")
    print(f"Train position: i={args.train_position}  (dataset i range: {i_range[0]} to {i_range[1]})")
    print(f"Probe type:     {args.probe_type}")
    print(f"Output dir:     {run_dir}")
    print("=" * 70)

    all_results = train_all_layers_at_position(
        train_position=args.train_position,
        train_dataset_path=args.train_dataset,
        val_dataset_path=args.val_dataset,
        probe_type=args.probe_type,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        output_dir=str(probes_dir),
        device=args.device,
        save_weights=args.save_weights,
        tokenizer=tokenizer,
    )

    # Save training summary
    summary_path = probes_dir / "training_summary_poem.pt"
    torch.save({
        'all_results': all_results,
        'train_position': args.train_position,
        'config': {
            'probe_type': args.probe_type,
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
        },
        'metadata': metadata,
    }, summary_path)

    # Write experiment_results.json
    config = {
        'model_name': metadata.get('model_name', 'unknown'),
        'task': 'poem_rhyme_prediction_i_indexed',
        'train_position': args.train_position,
        'train_dataset': args.train_dataset,
        'val_dataset': args.val_dataset,
        'probe_type': args.probe_type,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
    }

    results_data = make_experiment_results_json(all_results, metadata, config)
    results_file = run_dir / 'experiment_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total probes trained: {len(all_results)}")
    print(f"Results saved to:     {results_file}")
    print(f"Summary saved to:     {summary_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
