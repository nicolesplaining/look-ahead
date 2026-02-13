"""Build look-ahead activation dataset for probe training."""

import argparse
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

from .activation_extraction import generate_and_extract_all_layers
from .data_loading import load_jsonl_prompts


def main():
    parser = argparse.ArgumentParser(description="Build multi-layer activation dataset")

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--prompts_path", type=str, default=None)
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--split_field", type=str, default="split")
    parser.add_argument("--split_value", type=str, default=None)
    parser.add_argument("--max_prompts", type=int, default=None)

    parser.add_argument("--max_k", type=int, required=True,
                        help="Maximum lookahead distance (extracts targets for k=1,2,...,max_k)")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices or None for all layers")

    parser.add_argument("--output_path", type=str, required=True)

    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    model = HookedTransformer.from_pretrained(
        args.model_name, device=args.device, dtype=torch.bfloat16
    )

    if args.layers is not None:
        layers = [int(x.strip()) for x in args.layers.split(',')]
    else:
        layers = None

    print("Loading prompts...")
    if args.prompts_path is not None:
        prompts = load_jsonl_prompts(
            args.prompts_path, text_field=args.text_field,
            split_field=None, max_examples=args.max_prompts
        )
    elif args.dataset_path is not None:
        prompts = load_jsonl_prompts(
            args.dataset_path, text_field=args.text_field,
            split_field=args.split_field, split_value=args.split_value,
            max_examples=args.max_prompts
        )
    else:
        raise ValueError("Must provide --prompts_path or --dataset_path")

    print(f"Loaded {len(prompts)} prompts")

    print(f"Extracting activations (max_k={args.max_k})...")
    layer_activations, targets, generated_texts = generate_and_extract_all_layers(
        model=model, prompts=prompts, max_k=args.max_k,
        max_new_tokens=args.max_new_tokens, device=args.device, layers=layers
    )

    print(f"\nExample texts (first 3):")
    for i, text in enumerate(generated_texts[:3]):
        print(f"  {i+1}. {text[:100]}...")

    metadata = {
        'model_name': args.model_name,
        'max_k': args.max_k,
        'max_new_tokens': args.max_new_tokens,
        'n_prompts': len(prompts),
        'd_model': model.cfg.d_model,
        'vocab_size': model.cfg.d_vocab,
        'layers': layers if layers is not None else list(range(model.cfg.n_layers)),
    }

    dataset = {
        'layer_activations': layer_activations,
        'targets': targets,  # [n_samples, max_k]
        'generated_texts': generated_texts,
        'metadata': metadata
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)

    print(f"\nDataset saved to: {output_path}")
    print(f"Layers: {sorted(layer_activations.keys())}")
    print(f"Samples: {len(targets)}, Targets shape: {targets.shape}")


if __name__ == "__main__":
    main()
