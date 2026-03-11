#!/usr/bin/env python3
"""
Step 2: Apply steering vectors to validation examples and measure rhyme accuracy.

For each (src_scheme, tgt_scheme, layer, position_i):
  - Baseline: generate without steering; check if output rhymes with src scheme
  - Steered:  generate with vector[(src->tgt, layer, i)] injected at (layer, i);
              check if output rhymes with tgt scheme

For rel_pos <= 0: the vector computed at that prompt position is injected there.
For rel_pos >  0: the vector computed at --gen-vector-pos (default 0, the newline)
                  is injected at the rel_pos-th generated token.

Usage:
    python -m steering_probe.run_steering \
        --model <model_name_or_path> \
        --vectors-path <results/steering_vectors.pt> \
        --data-path <poems-val.jsonl> \
        --output-dir <results/> \
        [--layers 0 16 32] \
        [--positions -5 -3 -1 0 1 2 3] \
        [--alpha 20.0] \
        [--max-new-tokens 20] \
        [--source 0 1] [--target 2 3] \
        [--gen-vector-pos 0] \
        [--device cuda] [--dtype bfloat16]
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Run steering experiment on validation data")
    p.add_argument("--model", required=True)
    p.add_argument("--vectors-path", required=True, help="Path to steering_vectors.pt")
    p.add_argument("--data-path", required=True, help="Validation JSONL (fields: id, scheme, text)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--layers", nargs="+", type=int, default=None,
                   help="Layers to test (default: all in vectors file)")
    p.add_argument("--positions", nargs="+", type=int, default=None,
                   help="Positions i to test (default: all prompt positions in vectors file)")
    p.add_argument("--gen-positions", nargs="+", type=int, default=None,
                   help="Generation positions (i > 0) to test (default: none)")
    p.add_argument("--gen-vector-pos", type=int, default=0,
                   help="Which prompt position's vector to use for generation-position steering "
                        "(default: 0 = newline)")
    p.add_argument("--alpha", type=float, default=20.0)
    p.add_argument("--max-new-tokens", type=int, default=20)
    p.add_argument("--source", nargs="+", type=int, default=None,
                   help="Source scheme IDs (default: all)")
    p.add_argument("--target", nargs="+", type=int, default=None,
                   help="Target scheme IDs (default: all)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--normalize", action="store_true",
                   help="L2-normalize steering vectors before applying alpha")
    return p.parse_args()


def _save_results(results: dict, output_dir: str):
    """Save results incrementally so progress is not lost."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


def _print(msg: str = ""):
    """Print with immediate flush for SLURM log visibility."""
    print(msg, flush=True)


def main():
    args = parse_args()

    # --- load data ---
    examples = []
    with open(args.data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    _print(f"Loaded {len(examples)} val examples")

    by_scheme: Dict[int, List[dict]] = defaultdict(list)
    for ex in examples:
        by_scheme[int(ex["scheme"])].append(ex)

    # --- load vectors ---
    from .vectors import load_vectors
    vectors, scheme_means, metadata = load_vectors(args.vectors_path)
    _print(f"Loaded vectors for schemes: {metadata.get('schemes')}")

    # --- load model ---
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .steer import generate_baseline, generate_with_steering
    from .extract import find_last_newline_pos, get_newline_token_id
    from .evaluate import scheme_rhyme_key, evaluate_rhyme

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    _print(f"Loading model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device
    )
    model.eval()
    newline_id = get_newline_token_id(tokenizer)

    # --- determine layers and positions to sweep ---
    sample_key = next(iter(vectors))
    available_layers = sorted(vectors[sample_key].keys())
    layers = args.layers if args.layers is not None else available_layers

    # Prompt positions: from the vectors file
    sample_layer = available_layers[0]
    available_prompt_pos = sorted(vectors[sample_key][sample_layer].keys())
    prompt_positions = args.positions if args.positions is not None else available_prompt_pos

    gen_positions: List[int] = args.gen_positions or []

    # --- infer rhyme keys per scheme from val texts ---
    scheme_keys: Dict[int, Optional[str]] = {}
    for s, exs in by_scheme.items():
        scheme_keys[s] = scheme_rhyme_key([e["text"] for e in exs])
        _print(f"  Scheme {s} rhyme key: {scheme_keys[s]}")

    # --- determine which pairs to run ---
    all_schemes = sorted(by_scheme.keys())
    src_schemes = args.source if args.source is not None else all_schemes
    tgt_schemes = args.target if args.target is not None else all_schemes

    results: dict = {}

    for src in src_schemes:
        results[str(src)] = {}
        src_examples = by_scheme[src]
        if not src_examples:
            continue

        # Compute baseline once per source scheme
        _print(f"\n--- Baseline for scheme {src} ---")
        baseline_correct = sum(
            evaluate_rhyme(
                generate_baseline(model, tokenizer, ex["text"], args.max_new_tokens, args.device),
                scheme_keys[src],
            )
            for ex in src_examples
            if scheme_keys[src]
        )
        baseline_pct = baseline_correct / len(src_examples)
        _print(f"  baseline rhyme% = {baseline_pct:.1%}")

        for tgt in tgt_schemes:
            if tgt == src:
                continue
            if (src, tgt) not in vectors:
                _print(f"  No vector for ({src},{tgt}), skipping")
                continue
            if scheme_keys[tgt] is None:
                _print(f"  No rhyme key for scheme {tgt}, skipping")
                continue

            results[str(src)][str(tgt)] = {}
            _print(f"\n  Pair ({src} -> {tgt})")

            # --- prompt positions ---
            for layer in layers:
                if layer not in vectors[(src, tgt)]:
                    continue
                results[str(src)][str(tgt)][str(layer)] = {}

                for rel_pos in prompt_positions:
                    if rel_pos not in vectors[(src, tgt)][layer]:
                        continue
                    vec = vectors[(src, tgt)][layer][rel_pos]

                    correct = 0
                    for ex in src_examples:
                        toks = tokenizer(ex["text"], return_tensors="pt")["input_ids"][0]
                        npos = find_last_newline_pos(toks, newline_id)
                        gen = generate_with_steering(
                            model, tokenizer, ex["text"],
                            vec, layer, rel_pos, args.alpha,
                            args.max_new_tokens, args.device,
                            newline_pos=npos,
                            normalize=args.normalize,
                        )
                        if evaluate_rhyme(gen, scheme_keys[tgt]):
                            correct += 1

                    steered_pct = correct / len(src_examples)
                    results[str(src)][str(tgt)][str(layer)][str(rel_pos)] = {
                        "steered_rhyme_pct": steered_pct,
                        "baseline_rhyme_pct": baseline_pct,
                        "n": len(src_examples),
                    }
                    _print(f"    layer={layer:3d}  pos={rel_pos:4d}  "
                          f"steered={steered_pct:.1%}  baseline={baseline_pct:.1%}")
                    _save_results(results, args.output_dir)

            # --- generation positions (use gen_vector_pos vector) ---
            for gen_pos in gen_positions:
                if args.gen_vector_pos not in vectors[(src, tgt)].get(available_layers[0], {}):
                    _print(f"  gen_vector_pos={args.gen_vector_pos} not in vectors, skipping gen positions")
                    break
                for layer in layers:
                    if layer not in vectors[(src, tgt)]:
                        continue
                    if args.gen_vector_pos not in vectors[(src, tgt)][layer]:
                        continue
                    vec = vectors[(src, tgt)][layer][args.gen_vector_pos]

                    correct = 0
                    for ex in src_examples:
                        gen = generate_with_steering(
                            model, tokenizer, ex["text"],
                            vec, layer, gen_pos, args.alpha,
                            args.max_new_tokens, args.device,
                            newline_pos=None,  # not needed for gen positions
                            normalize=args.normalize,
                        )
                        if evaluate_rhyme(gen, scheme_keys[tgt]):
                            correct += 1

                    steered_pct = correct / len(src_examples)
                    lkey = str(layer)
                    if lkey not in results[str(src)][str(tgt)]:
                        results[str(src)][str(tgt)][lkey] = {}
                    results[str(src)][str(tgt)][lkey][str(gen_pos)] = {
                        "steered_rhyme_pct": steered_pct,
                        "baseline_rhyme_pct": baseline_pct,
                        "n": len(src_examples),
                        "gen_vector_pos": args.gen_vector_pos,
                    }
                    _print(f"    layer={layer:3d}  gen_pos={gen_pos:3d}  "
                          f"steered={steered_pct:.1%}  baseline={baseline_pct:.1%}")
                    _save_results(results, args.output_dir)

    # --- save ---
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    _print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
