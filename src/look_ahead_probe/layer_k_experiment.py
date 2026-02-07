#!/usr/bin/env python
"""
Layer-K probing experiment pipeline.

Orchestrates the 3-step pipeline:
1. Check model compatibility
2. Build activation datasets (train + optional validation)
3. Train & evaluate probes for all layers and k values

Results (train + val metrics) are saved to JSON for later analysis.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


def run_step(step_name, cmd):
    """Run a pipeline step and handle errors."""
    print("\n" + "=" * 80)
    print(f"{step_name}")
    print("=" * 80)
    print(f"Running: {' '.join(cmd)}")
    print("=" * 80 + "\n")

    result = subprocess.run(cmd, check=True)
    print(f"\n✓ {step_name} complete\n")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Layer-K probing experiment pipeline"
    )

    # Model and data
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model to probe")
    parser.add_argument("--train_dataset_path", type=str, required=True,
                        help="Path to JSONL training dataset")
    parser.add_argument("--val_dataset_path", type=str, default=None,
                        help="Path to JSONL validation dataset (optional)")
    parser.add_argument("--max_k", type=int, default=3,
                        help="Maximum lookahead distance")

    # Optional parameters
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Max prompts to use")
    parser.add_argument("--max_new_tokens", type=int, default=30,
                        help="Max tokens to generate per prompt")

    # Training
    parser.add_argument("--probe_type", type=str, default="linear",
                        choices=["linear", "mlp"],
                        help="Probe architecture")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate")

    # Output
    parser.add_argument("--output_dir", type=str, default="experiment_results",
                        help="Output directory")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device")
    parser.add_argument("--skip_check", action="store_true",
                        help="Skip model compatibility check")

    args = parser.parse_args()

    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / "activations.pt"
    val_dataset_path = output_dir / "val_activations.pt" if args.val_dataset_path else None
    probes_dir = output_dir / "probes"
    probes_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 80)
    print("LAYER-K PROBING EXPERIMENT PIPELINE")
    print("=" * 80)
    print(f"Model:      {args.model_name}")
    print(f"Dataset:    {args.train_dataset_path}")
    print(f"K range:    1 to {args.max_k}")
    print(f"Probe:      {args.probe_type}")
    print(f"Output:     {args.output_dir}")
    print("=" * 80)

    # =========================================================================
    # STEP 1: Check Model Compatibility
    # =========================================================================
    # Verifies that the model's causal masking allows efficient activation
    # extraction (i.e., activation at position i only depends on tokens 0..i)

    if not args.skip_check:
        check_cmd = [
            "python", "-m", "look_ahead_probe.check_model",
            "--model_name", args.model_name,
            "--device", args.device,
            "--max_new_tokens", "10",  # Quick check
        ]
        run_step("STEP 1: Check Model Compatibility", check_cmd)
    else:
        print("\n⚠️  Skipping model check (--skip_check enabled)\n")

    # =========================================================================
    # STEP 2: Build Activation Dataset
    # =========================================================================
    # Extracts activations from all layers for k=1,2,...,max_k
    # Saves to a single .pt file for efficient reuse

    build_cmd = [
        "python", "-m", "look_ahead_probe.build_look_ahead_activation_dataset",
        "--model_name", args.model_name,
        "--prompts_path", args.train_dataset_path,  # Use prompts_path to skip split filtering
        "--max_k", str(args.max_k),
        "--max_new_tokens", str(args.max_new_tokens),
        "--output_path", str(dataset_path),
        "--device", args.device,
    ]

    if args.max_prompts:
        build_cmd.extend(["--max_prompts", str(args.max_prompts)])

    run_step("STEP 2: Build Activation Dataset (Training)", build_cmd)

    # Build validation dataset if provided
    if args.val_dataset_path:
        val_build_cmd = [
            "python", "-m", "look_ahead_probe.build_look_ahead_activation_dataset",
            "--model_name", args.model_name,
            "--prompts_path", args.val_dataset_path,  # Use prompts_path to skip split filtering
            "--max_k", str(args.max_k),
            "--max_new_tokens", str(args.max_new_tokens),
            "--output_path", str(val_dataset_path),
            "--device", args.device,
        ]
        if args.max_prompts:
            val_build_cmd.extend(["--max_prompts", str(args.max_prompts)])

        run_step("STEP 2b: Build Activation Dataset (Validation)", val_build_cmd)

    # Load metadata for step 3
    data = torch.load(dataset_path, weights_only=False)
    metadata = data['metadata']

    print(f"\nDataset metadata:")
    print(f"  Layers: {len(metadata['layers'])}")
    print(f"  Max k: {metadata['max_k']}")
    print(f"  Model: {metadata['model_name']}")
    if args.val_dataset_path:
        print(f"  Validation dataset: ✓")

    # =========================================================================
    # STEP 3: Train Probes for All Layers and K Values
    # =========================================================================
    # For each k value, trains probes on all layers
    # Uses train_all_layers which is more efficient than training individually
    # If validation dataset provided, automatically evaluates and includes val metrics

    all_results = {}

    for k in range(1, args.max_k + 1):
        k_output_dir = probes_dir / f"k{k}"

        train_cmd = [
            "python", "-m", "look_ahead_probe.train_all_layers",
            "--train_dataset", str(dataset_path),
            "--k", str(k),
            "--probe_type", args.probe_type,
            "--num_epochs", str(args.num_epochs),
            "--batch_size", str(args.batch_size),
            "--learning_rate", str(args.learning_rate),
            "--output_dir", str(k_output_dir),
            "--device", args.device,
        ]

        if args.val_dataset_path:
            train_cmd.extend(["--val_dataset", str(val_dataset_path)])

        run_step(f"STEP 3.{k}: Train All Layers for k={k}", train_cmd)

        # Load training results
        summary_path = k_output_dir / f"training_summary_k{k}.pt"
        print(f"Looking for summary at: {summary_path}")

        if summary_path.exists():
            print(f"✓ Found summary file")
            summary = torch.load(summary_path, weights_only=False)
            print(f"  Layers in summary: {list(summary['all_results'].keys())}")

            # Extract results for each layer
            for layer_idx, layer_data in summary['all_results'].items():
                # Check if validation results exist
                if layer_data.get('results') is not None:
                    # Has validation results
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
                    # No validation, use final training metrics from history
                    all_results[f"layer{layer_idx}_k{k}"] = {
                        'layer': layer_idx,
                        'k': k,
                        'train_accuracy': float(layer_data['history']['train_acc'][-1]),
                        'train_loss': float(layer_data['history']['train_loss'][-1]),
                        'probe_path': layer_data['save_path'],
                    }
                else:
                    print(f"    ⚠️  No results or history for layer {layer_idx}")

            print(f"  Extracted {len([k for k in all_results if f'_k{k}' in k])} results for k={k}")
        else:
            print(f"✗ Summary file not found!")
            print(f"  Check if training completed successfully")

    # =========================================================================
    # Save Aggregated Results
    # =========================================================================
    # Note: Validation results are already included from train_all_layers if
    # --val_dataset_path was provided
    # Combines results from all k values into a single JSON file for easy
    # analysis and visualization

    if not all_results:
        print("\n⚠️  WARNING: No results collected!")
        print("This usually means training summaries weren't found.")
        print("Check that train_all_layers completed successfully.")

    results_file = output_dir / 'experiment_results.json'

    results_data = {
        'config': {
            'model_name': args.model_name,
            'dataset_path': args.train_dataset_path,
            'max_k': args.max_k,
            'probe_type': args.probe_type,
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
        },
        'metadata': metadata,
        'results': all_results,
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Total probes trained: {len(all_results)}")
    print(f"Results saved to: {results_file}")
    print(f"Probes saved in: {probes_dir}")
    print("=" * 80 + "\n")

    # Print summary statistics
    print("Summary Statistics:")
    print("-" * 80)
    for k in range(1, args.max_k + 1):
        k_results = [v for key, v in all_results.items() if v['k'] == k]
        if k_results:
            # Training stats
            train_accs = [r['train_accuracy'] for r in k_results if 'train_accuracy' in r]
            if train_accs:
                best_train = max(k_results, key=lambda x: x.get('train_accuracy', 0))
                avg_train = sum(train_accs) / len(train_accs)
                print(f"k={k} (Train): Best=Layer {best_train['layer']} ({best_train['train_accuracy']:.2%}), "
                      f"Average={avg_train:.2%}")

            # Validation stats if available
            val_accs = [r['val_accuracy'] for r in k_results if 'val_accuracy' in r]
            if val_accs:
                best_val = max(k_results, key=lambda x: x.get('val_accuracy', 0))
                avg_val = sum(val_accs) / len(val_accs)
                print(f"k={k} (Val):   Best=Layer {best_val['layer']} ({best_val['val_accuracy']:.2%}), "
                      f"Average={avg_val:.2%}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
