#!/usr/bin/env python3
"""
Step 1: Extract activations from the training set and compute pairwise steering vectors.

Usage:
    python -m steering_probe.compute_vectors \
        --model <model_name_or_path> \
        --data-path <poems-train.jsonl> \
        --output-dir <results/> \
        [--context-window 20] \
        [--layers 0 16 32 48] \
        [--device cuda] \
        [--dtype bfloat16]
"""
import argparse
import json
import os

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Compute steering vectors from training data")
    p.add_argument("--model", required=True, help="Model name or path")
    p.add_argument("--data-path", required=True, help="Training JSONL (fields: id, scheme, text)")
    p.add_argument("--output-dir", required=True, help="Directory to write steering_vectors.pt")
    p.add_argument("--context-window", type=int, default=20,
                   help="# tokens before newline to extract activations for (default: 20)")
    p.add_argument("--layers", nargs="+", type=int, default=None,
                   help="Layer indices to use; omit for all layers")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    return p.parse_args()


def main():
    args = parse_args()

    examples = []
    with open(args.data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    print(f"Loaded {len(examples)} training examples")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .extract import extract_scheme_means
    from .vectors import compute_steering_vectors, save_vectors

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    print(f"Loading model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device
    )
    model.eval()

    from .extract import get_layers
    n_layers = len(get_layers(model))
    layers = args.layers if args.layers is not None else list(range(n_layers))
    print(f"Model: {n_layers} layers, hidden_size={model.config.hidden_size}")
    print(f"Sweeping {len(layers)} layers, context_window={args.context_window}")

    print("Extracting activations ...")
    scheme_means = extract_scheme_means(
        model, tokenizer, examples, args.context_window, layers, args.device
    )

    schemes = sorted(scheme_means.keys())
    print(f"Schemes found: {schemes}")

    print("Computing pairwise steering vectors ...")
    vectors = compute_steering_vectors(scheme_means)
    print(f"Computed {len(vectors)} directional pairs")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "steering_vectors.pt")
    metadata = {
        "model": args.model,
        "context_window": args.context_window,
        "layers": layers,
        "schemes": schemes,
        "n_examples": len(examples),
    }
    save_vectors(vectors, scheme_means, out_path, metadata)
    print(f"Saved steering vectors → {out_path}")


if __name__ == "__main__":
    main()
