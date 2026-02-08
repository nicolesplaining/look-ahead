#!/usr/bin/env python3
"""
Create training and validation datasets from The Pile.

Samples random token sequences from The Pile and saves them as JSONL files
suitable for the look-ahead probe pipeline.

Note: Tokenizer is used to measure and sample sequences by TOKEN length (not
characters), ensuring consistent behavior during generation. Output is saved
as text strings, not tokens - the text gets tokenized again during the pipeline.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


def sample_sequences_from_pile(
    dataset_name: str,
    tokenizer,
    n_sequences: int,
    min_tokens: int = 64,
    max_tokens: int = 256,
    seed: int = 42,
    subset: str = None
) -> List[str]:
    """
    Sample random token sequences from The Pile.

    Args:
        dataset_name: HuggingFace dataset name
        tokenizer: Tokenizer to use for encoding/decoding
        n_sequences: Number of sequences to sample
        min_tokens: Minimum sequence length in tokens
        max_tokens: Maximum sequence length in tokens
        seed: Random seed for reproducibility
        subset: Specific subset of The Pile to use (None = all)

    Returns:
        List of text sequences
    """
    print(f"\nLoading dataset: {dataset_name}")
    if subset:
        print(f"  Subset: {subset}")
    print(f"  Target sequences: {n_sequences}")
    print(f"  Token length: {min_tokens}-{max_tokens}")

    # Load dataset in streaming mode (doesn't download everything)
    try:
        if subset:
            dataset = load_dataset(
                dataset_name,
                subset,
                split="train",
                streaming=True,
                trust_remote_code=True
            )
        else:
            dataset = load_dataset(
                dataset_name,
                split="train",
                streaming=True,
                trust_remote_code=True
            )
    except Exception as e:
        print(f"\n⚠️  Warning: Could not load with streaming. Trying without...")
        if subset:
            dataset = load_dataset(
                dataset_name,
                subset,
                split="train",
                trust_remote_code=True
            )
        else:
            dataset = load_dataset(
                dataset_name,
                split="train",
                trust_remote_code=True
            )

    sequences = []
    random.seed(seed)

    # Shuffle and iterate
    dataset_shuffled = dataset.shuffle(seed=seed, buffer_size=10000)

    print("\nSampling sequences...")
    with tqdm(total=n_sequences, desc="Sampling") as pbar:
        for example in dataset_shuffled:
            if len(sequences) >= n_sequences:
                break

            # Get text from example
            if isinstance(example, dict):
                text = example.get('text', example.get('content', ''))
            else:
                text = str(example)

            if not text or len(text.strip()) < 50:
                continue

            # Tokenize to measure length in tokens (not chars)
            # Output is saved as text, not tokens - this ensures consistent token lengths
            try:
                tokens = tokenizer.encode(text, add_special_tokens=False)
            except Exception as e:
                continue

            # Check if long enough to sample from
            if len(tokens) < min_tokens:
                continue

            # Sample a random subsequence of specific token length
            seq_length = random.randint(min_tokens, min(max_tokens, len(tokens)))

            if len(tokens) >= seq_length:
                start = random.randint(0, len(tokens) - seq_length)
                token_seq = tokens[start:start + seq_length]

                # Decode back to text for storage (saved as strings, not token IDs)
                try:
                    sampled_text = tokenizer.decode(token_seq, skip_special_tokens=True)

                    # Quality check: make sure it's not empty after decoding
                    if sampled_text and len(sampled_text.strip()) > 20:
                        sequences.append(sampled_text)
                        pbar.update(1)
                except Exception as e:
                    continue

    print(f"✓ Sampled {len(sequences)} sequences")
    return sequences


def save_jsonl(sequences: List[str], output_path: Path):
    """Save sequences as JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for seq in sequences:
            json.dump({"text": seq}, f, ensure_ascii=False)
            f.write('\n')

    print(f"✓ Saved {len(sequences)} sequences to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create training datasets from The Pile"
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="monology/pile-uncopyrighted",
        help="HuggingFace dataset name (default: monology/pile-uncopyrighted)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Model name for tokenizer (default: meta-llama/Llama-3.2-1B)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory (default: data/)"
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=10000,
        help="Number of training sequences (default: 10000)"
    )
    parser.add_argument(
        "--n_val",
        type=int,
        default=2000,
        help="Number of validation sequences (default: 2000)"
    )
    parser.add_argument(
        "--n_small_train",
        type=int,
        default=50,
        help="Number of small training sequences (default: 50)"
    )
    parser.add_argument(
        "--n_small_val",
        type=int,
        default=10,
        help="Number of small validation sequences (default: 10)"
    )
    parser.add_argument(
        "--min_tokens",
        type=int,
        default=64,
        help="Minimum sequence length in tokens (default: 64)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum sequence length in tokens (default: 256)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Specific Pile subset to use (e.g., 'Wikipedia_(en)')"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("=" * 80)
    print("PILE DATASET CREATION")
    print("=" * 80)
    print(f"Dataset: {args.dataset_name}")
    print(f"Tokenizer: {args.model_name}")
    print(f"Output directory: {output_dir}")
    print(f"Training sequences: {args.n_train}")
    print(f"Validation sequences: {args.n_val}")
    print(f"Small train/val: {args.n_small_train}/{args.n_small_val}")
    print(f"Token range: {args.min_tokens}-{args.max_tokens}")
    print(f"Random seed: {args.seed}")
    print("=" * 80)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(f"✓ Loaded tokenizer for {args.model_name}")

    # Sample training sequences
    print("\n" + "=" * 80)
    print("TRAINING SET")
    print("=" * 80)
    train_sequences = sample_sequences_from_pile(
        args.dataset_name,
        tokenizer,
        n_sequences=args.n_train,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        seed=args.seed,
        subset=args.subset
    )

    # Sample validation sequences (different seed)
    print("\n" + "=" * 80)
    print("VALIDATION SET")
    print("=" * 80)
    val_sequences = sample_sequences_from_pile(
        args.dataset_name,
        tokenizer,
        n_sequences=args.n_val,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        seed=args.seed + 1,  # Different seed for val
        subset=args.subset
    )

    # Save full datasets
    print("\n" + "=" * 80)
    print("SAVING DATASETS")
    print("=" * 80)

    save_jsonl(train_sequences, output_dir / "train-pile.jsonl")
    save_jsonl(val_sequences, output_dir / "val-pile.jsonl")

    # Create small datasets (first N samples)
    small_train = train_sequences[:args.n_small_train]
    small_val = val_sequences[:args.n_small_val]

    save_jsonl(small_train, output_dir / "small-train-pile.jsonl")
    save_jsonl(small_val, output_dir / "small-val-pile.jsonl")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Created 4 dataset files in {output_dir}/")
    print(f"  - train-pile.jsonl ({len(train_sequences):,} sequences)")
    print(f"  - val-pile.jsonl ({len(val_sequences):,} sequences)")
    print(f"  - small-train-pile.jsonl ({len(small_train):,} sequences)")
    print(f"  - small-val-pile.jsonl ({len(small_val):,} sequences)")
    print(f"\nExample sequence (first 200 chars):")
    print(f"  {train_sequences[0][:200]}...")
    print("=" * 80)


if __name__ == "__main__":
    main()
