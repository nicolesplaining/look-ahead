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
    # Print Summary
    # =========================================================================

    # Load results
    summary_path = probes_dir / "training_summary.pt"
    if summary_path.exists():
        summary = torch.load(summary_path, weights_only=False)

        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETE")
        print("=" * 80)
        print(f"Results saved to: {args.output_dir}")
        print("\nAccuracy by Layer:")
        print("-" * 80)

        for key, data in sorted(summary['all_results'].items()):
            layer = data['layer']
            train_acc = data['history']['train_acc'][-1]
            if data['results']:
                val_acc = data['results']['accuracy']
                print(f"  Layer {layer:2d}: Train={train_acc:.2%}, Val={val_acc:.2%}")
            else:
                print(f"  Layer {layer:2d}: Train={train_acc:.2%}")

        print("=" * 80)


if __name__ == "__main__":
    main()
