#!/usr/bin/env python3
"""
Export generated_texts from an activations .pt file to a JSONL.

Usage:
    python -m export_texts poem/data/activations_train.pt
    python -m export_texts poem/data/activations_train.pt --output poem/data/generated_train.jsonl
"""

import argparse
import json
import torch
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export generated_texts from a .pt dataset to JSONL")
    parser.add_argument("pt_path", type=str, help="Path to activations .pt file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path (default: same location as .pt, with .jsonl extension)")
    args = parser.parse_args()

    pt_path = Path(args.pt_path)
    out_path = Path(args.output) if args.output else pt_path.with_suffix(".jsonl")

    print(f"Loading {pt_path} ...")
    data = torch.load(pt_path, weights_only=False)

    texts = data.get("generated_texts", [])
    metadata = data.get("metadata", {})

    print(f"Found {len(texts)} generated texts")
    print(f"Model: {metadata.get('model_name', 'unknown')}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for i, text in enumerate(texts):
            f.write(json.dumps({"id": i, "text": text}) + "\n")

    print(f"✓ Written to {out_path}")


if __name__ == "__main__":
    main()
