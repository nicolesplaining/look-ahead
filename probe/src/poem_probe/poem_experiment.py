#!/usr/bin/env python3
"""
Poem rhyme probe experiment pipeline.

Orchestrates the full pipeline:
1. Split poems into train/val
2. Extract activation datasets
3. Train probes on all layers
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


def split_poems(poems_path: Path, train_path: Path, val_path: Path, train_ratio: float = 0.8):
    """Split poems into train/val sets."""
    print(f"Splitting poems: {poems_path}")

    # Read all poems
    poems = []
    with open(poems_path, 'r') as f:
        for line in f:
            poems.append(json.loads(line))

    # Split
    n_train = int(len(poems) * train_ratio)
    train_poems = poems[:n_train]
    val_poems = poems[n_train:]

    # Save
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)

    with open(train_path, 'w') as f:
        for poem in train_poems:
            f.write(json.dumps(poem) + '\n')

    with open(val_path, 'w') as f:
        for poem in val_poems:
            f.write(json.dumps(poem) + '\n')

    print(f"  Train: {len(train_poems)} poems → {train_path}")
    print(f"  Val:   {len(val_poems)} poems → {val_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Poem rhyme probe experiment pipeline"
    )

    # Data
    parser.add_argument("--poems_path", type=str, required=True,
                        help="Path to poems JSONL file")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model to probe")

    # Extraction
    parser.add_argument("--max_new_tokens", type=int, default=30,
                        help="Max tokens to generate per poem")

    # Training
    parser.add_argument("--probe_type", type=str, default="mlp",
                        choices=["linear", "mlp"],
                        help="Probe architecture")
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate")

    # Output
    parser.add_argument("--output_dir", type=str, default="poem_results",
                        help="Output directory")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (None = all)")

    args = parser.parse_args()

    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    train_poems_path = data_dir / "poems_train.jsonl"
    val_poems_path = data_dir / "poems_val.jsonl"
    train_dataset_path = data_dir / "train.pt"
    val_dataset_path = data_dir / "val.pt"
    probes_dir = output_dir / "probes"

    print("\n" + "=" * 80)
    print("POEM RHYME PROBE EXPERIMENT PIPELINE")
    print("=" * 80)
    print(f"Model:      {args.model_name}")
    print(f"Poems:      {args.poems_path}")
    print(f"Probe:      {args.probe_type}")
    print(f"Epochs:     {args.num_epochs}")
    print(f"Output:     {args.output_dir}")
    print("=" * 80)

    # =========================================================================
    # STEP 1: Split Poems into Train/Val
    # =========================================================================

    if not train_poems_path.exists() or not val_poems_path.exists():
        run_step(
            "STEP 1: Split Poems into Train/Val",
            ["echo", "Splitting poems..."]
        )
        split_poems(
            Path(args.poems_path),
            train_poems_path,
            val_poems_path,
            train_ratio=0.8
        )
    else:
        print("\n⚠️  Using existing train/val splits\n")

    # =========================================================================
    # STEP 2: Extract Training Dataset
    # =========================================================================

    extract_train_cmd = [
        "python", "-m", "poem_probe.extract_poem_dataset",
        "--model_name", args.model_name,
        "--poems_path", str(train_poems_path),
        "--output_path", str(train_dataset_path),
        "--max_new_tokens", str(args.max_new_tokens),
        "--device", args.device,
    ]

    if args.layers:
        extract_train_cmd.extend(["--layers", args.layers])

    run_step("STEP 2: Extract Training Dataset", extract_train_cmd)

    # =========================================================================
    # STEP 3: Extract Validation Dataset
    # =========================================================================

    extract_val_cmd = [
        "python", "-m", "poem_probe.extract_poem_dataset",
        "--model_name", args.model_name,
        "--poems_path", str(val_poems_path),
        "--output_path", str(val_dataset_path),
        "--max_new_tokens", str(args.max_new_tokens),
        "--device", args.device,
    ]

    if args.layers:
        extract_val_cmd.extend(["--layers", args.layers])

    run_step("STEP 3: Extract Validation Dataset", extract_val_cmd)

    # =========================================================================
    # STEP 4: Train Probes on All Layers
    # =========================================================================

    train_cmd = [
        "python", "-m", "poem_probe.train_poem_probe",
        "--train_dataset", str(train_dataset_path),
        "--val_dataset", str(val_dataset_path),
        "--probe_type", args.probe_type,
        "--num_epochs", str(args.num_epochs),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--output_dir", str(probes_dir),
        "--device", args.device,
    ]

    if args.layers:
        train_cmd.extend(["--layers", args.layers])

    run_step("STEP 4: Train Probes", train_cmd)

    # =========================================================================
    # Save Results and Generated Poems
    # =========================================================================

    # Load training results
    summary_path = probes_dir / "training_summary.pt"
    all_results = {}

    if summary_path.exists():
        summary = torch.load(summary_path, weights_only=False)

        # Extract results for each layer
        for key, data in summary['all_results'].items():
            layer = data['layer']

            result_entry = {
                'layer': layer,
                'train_accuracy': float(data['history']['train_acc'][-1]),
                'train_loss': float(data['history']['train_loss'][-1]),
                'probe_path': data['save_path'],
            }

            # Add validation metrics if available
            if data['results']:
                result_entry['val_accuracy'] = float(data['results']['accuracy'])
                result_entry['val_top5_accuracy'] = float(data['results']['top5_accuracy'])
                result_entry['val_loss'] = float(data['results']['loss'])

            all_results[f"layer{layer}"] = result_entry

    # Load generated poems from datasets
    train_data = torch.load(train_dataset_path, weights_only=False)
    val_data = torch.load(val_dataset_path, weights_only=False)

    generated_poems = {
        'train': train_data['generated_texts'],
        'val': val_data['generated_texts'],
    }

    # Save results to JSON
    results_file = output_dir / "experiment_results.json"
    results_data = {
        'config': {
            'model_name': args.model_name,
            'poems_path': args.poems_path,
            'probe_type': args.probe_type,
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'max_new_tokens': args.max_new_tokens,
        },
        'metadata': {
            'n_train_poems': len(train_data['generated_texts']),
            'n_val_poems': len(val_data['generated_texts']),
            'd_model': train_data['metadata']['d_model'],
            'vocab_size': train_data['metadata']['vocab_size'],
            'layers': train_data['metadata']['layers'],
        },
        'results': all_results,
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    # Save generated poems to JSON
    poems_file = output_dir / "generated_poems.json"
    poems_data = {
        'train_poems': generated_poems['train'],
        'val_poems': generated_poems['val'],
        'n_train': len(generated_poems['train']),
        'n_val': len(generated_poems['val']),
    }

    with open(poems_file, 'w') as f:
        json.dump(poems_data, f, indent=2)

    # Print Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print(f"  - Probes: {probes_dir}")
    print(f"  - Results JSON: {results_file}")
    print(f"  - Generated poems JSON: {poems_file}")
    print("\nAccuracy by Layer:")
    print("-" * 80)

    for key, data in sorted(all_results.items()):
        layer = data['layer']
        train_acc = data['train_accuracy']
        if 'val_accuracy' in data:
            val_acc = data['val_accuracy']
            print(f"  Layer {layer:2d}: Train={train_acc:.2%}, Val={val_acc:.2%}")
        else:
            print(f"  Layer {layer:2d}: Train={train_acc:.2%}")

    print("-" * 80)
    print(f"Generated {len(generated_poems['train'])} train poems, {len(generated_poems['val'])} val poems")
    print("=" * 80)


if __name__ == "__main__":
    main()
