#!/usr/bin/env python3
"""
Extract activations and targets for poem rhyme prediction.

For each poem prompt (incomplete couplet), generates the second line and creates:
- Input:  activation at the last token of the FIRST line (end of prompt)
- Target: the last token of the second line BEFORE the terminating newline
          (i.e. the rhyming word)

Generation stops as soon as a newline token is emitted.  Examples where no
newline is generated within max_new_tokens are skipped.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm


def load_poem_prompts(jsonl_path: str, max_prompts: Optional[int] = None) -> List[str]:
    """Load poem prompts from JSONL file."""
    import json
    prompts = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data['text'])
            if max_prompts is not None and len(prompts) >= max_prompts:
                break
    return prompts


def extract_poem_activations(
    model: HookedTransformer,
    prompts: List[str],
    max_new_tokens: int = 16,
    device: str = "cuda",
    layers: Optional[List[int]] = None,
    chunk_size: int = 128,
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, List[str]]:
    """
    Extract activations and targets for poem rhyme prediction.

    For each poem:
    1. Generate tokens one at a time until a newline is produced (or max_new_tokens).
    2. Skip the example if no newline is generated.
    3. Target = the token immediately before the newline (the rhyming word).
    4. Activation = residual stream at the last prompt token (end of first line).

    Args:
        model: Language model
        prompts: Incomplete poem prompts (first line already included)
        max_new_tokens: Maximum tokens to generate before giving up
        device: Device to use
        layers: Layer indices to extract (None = all layers)
        chunk_size: Consolidate 1-D activation tensors into 2-D chunks every
            this many kept examples to bound CPU RAM usage.

    Returns:
        layer_activations: Dict mapping layer_idx -> [n_kept, d_model]
        targets:           [n_kept]   token IDs of the rhyming words
        generated_texts:   List of full generated couplet strings (n_kept entries)
    """
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    # Completed 2-D chunks (consolidated every chunk_size kept examples)
    layer_act_chunks = {layer_idx: [] for layer_idx in layers}
    # Current-chunk accumulation buffer (1-D tensors)
    layer_act_buf = {layer_idx: [] for layer_idx in layers}

    all_targets: List[torch.Tensor] = []
    generated_texts: List[str] = []

    model.eval()
    eos_token_id = model.tokenizer.eos_token_id
    n_skipped = 0
    kept_count = 0

    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Extracting poem activations"):
            prompt_tokens = model.to_tokens(prompt).to(device)
            prompt_length = prompt_tokens.shape[1]
            current_tokens = prompt_tokens.clone()

            # Generate token-by-token; stop as soon as a newline appears
            newline_pos: Optional[int] = None  # absolute token index of the \n token
            for step in range(max_new_tokens):
                logits = model(current_tokens)
                next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)
                token_text = model.tokenizer.decode([next_token.item()])
                current_tokens = torch.cat(
                    [current_tokens, next_token.unsqueeze(0)], dim=1
                )

                if '\n' in token_text:
                    newline_pos = prompt_length + step  # absolute index
                    break

                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break

            # Skip if no newline was generated
            if newline_pos is None:
                n_skipped += 1
                continue

            # Target: last token before the newline (the rhyming word)
            target_idx = newline_pos - 1
            if target_idx < prompt_length:
                # Newline was the very first generated token – nothing to probe
                n_skipped += 1
                continue

            # Single forward pass for activations
            _, cache = model.run_with_cache(
                current_tokens,
                names_filter=lambda name: name.endswith("hook_resid_post"),
            )

            target_token = current_tokens[0, target_idx]
            all_targets.append(target_token.cpu())
            generated_texts.append(model.tokenizer.decode(current_tokens[0]))

            # Activation at the last token of the prompt (end of first line)
            activation_position = prompt_length - 1
            for layer_idx in layers:
                layer_acts = cache["resid_post", layer_idx][0]
                act = layer_acts[activation_position, :]
                layer_act_buf[layer_idx].append(act.cpu())

            # Free GPU memory promptly
            del cache
            torch.cuda.empty_cache()

            kept_count += 1

            # Periodic consolidation to keep RAM bounded
            if kept_count % chunk_size == 0:
                for layer_idx in layers:
                    if layer_act_buf[layer_idx]:
                        layer_act_chunks[layer_idx].append(
                            torch.stack(layer_act_buf[layer_idx])
                        )
                        layer_act_buf[layer_idx] = []

    # Consolidate remaining buffer
    for layer_idx in layers:
        if layer_act_buf[layer_idx]:
            layer_act_chunks[layer_idx].append(
                torch.stack(layer_act_buf[layer_idx])
            )

    if n_skipped:
        print(f"Skipped {n_skipped} examples (no newline generated within {max_new_tokens} tokens)")

    layer_activations = {
        layer_idx: torch.cat(layer_act_chunks[layer_idx])
        for layer_idx in layers
    }
    targets = torch.stack(all_targets)  # [n_kept]

    print(f"\n✓ Extracted {kept_count} samples")
    print(f"  Layers: {sorted(layer_activations.keys())}")
    print(f"  Activation shape per layer: {list(layer_activations.values())[0].shape}")
    print(f"  Targets shape: {targets.shape}")

    return layer_activations, targets, generated_texts


def main():
    parser = argparse.ArgumentParser(
        description="Extract poem rhyme activations and targets"
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--poems_path", type=str, required=True,
                        help="Path to poems JSONL file")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save dataset (.pt)")
    parser.add_argument("--max_new_tokens", type=int, default=16,
                        help="Max tokens to generate per poem before skipping")
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Limit number of prompts loaded (for quick tests)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (None = all)")

    args = parser.parse_args()

    layers = None
    if args.layers is not None:
        layers = [int(x.strip()) for x in args.layers.split(',')]

    print("=" * 80)
    print("POEM RHYME ACTIVATION EXTRACTION")
    print("=" * 80)
    print(f"Model:          {args.model_name}")
    print(f"Poems:          {args.poems_path}")
    print(f"Output:         {args.output_path}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"Device:         {args.device}")
    print("=" * 80 + "\n")

    print("Loading model...")
    model = HookedTransformer.from_pretrained(
        args.model_name, device=args.device, dtype=torch.bfloat16
    )
    print(f"✓ Loaded {args.model_name}  (layers={model.cfg.n_layers}, d_model={model.cfg.d_model})\n")

    print("Loading poems...")
    prompts = load_poem_prompts(args.poems_path, max_prompts=args.max_prompts)
    print(f"✓ Loaded {len(prompts)} poem prompts\n")

    layer_activations, targets, generated_texts = extract_poem_activations(
        model=model,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        layers=layers,
    )

    # Show a few examples
    print("\nExample generations (first 3):")
    for i, text in enumerate(generated_texts[:3]):
        target_text = model.tokenizer.decode([targets[i].item()])
        print(f"\n{i+1}. {text[:150]}")
        print(f"   Rhyme target token: {repr(target_text)}")

    metadata = {
        'model_name': args.model_name,
        'n_samples': len(targets),
        'd_model': model.cfg.d_model,
        'vocab_size': model.cfg.d_vocab,
        'layers': layers if layers is not None else list(range(model.cfg.n_layers)),
        'task': 'poem_rhyme_prediction',
        'description': 'Predict rhyming word from activation at end of first line',
        'max_new_tokens': args.max_new_tokens,
    }

    dataset = {
        'layer_activations': layer_activations,
        'targets': targets,
        'generated_texts': generated_texts,
        'metadata': metadata,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)

    print(f"\n{'='*80}")
    print("DATASET SAVED")
    print("=" * 80)
    print(f"Path:    {output_path}")
    print(f"Samples: {len(targets)}")
    print(f"Layers:  {sorted(layer_activations.keys())}")
    print("=" * 80)


if __name__ == "__main__":
    main()
