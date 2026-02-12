#!/usr/bin/env python3
"""
Extract activations and targets for poem rhyme prediction.

For each poem prompt (incomplete couplet), generates the completion and creates:
- Input: Activation at the last token of the first line
- Target: The last token of the second line (the rhyming word)
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm


def load_poem_prompts(jsonl_path: str) -> List[str]:
    """Load poem prompts from JSONL file."""
    prompts = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data['text'])
    return prompts


def find_second_line_end(tokens: torch.Tensor, tokenizer, prompt_length: int) -> int:
    """
    Find where the second line ends in the generated sequence.

    Strategy: Look for newline after the first newline in generated text,
    or use punctuation as fallback.

    Args:
        tokens: Full token sequence [1, seq_len]
        tokenizer: Tokenizer for decoding
        prompt_length: Number of tokens in the original prompt

    Returns:
        Index of the last token of the second line
    """
    # Decode generated portion
    generated_tokens = tokens[0, prompt_length:]
    generated_text = tokenizer.decode(generated_tokens)

    # Count newlines - second line should end after we see another newline
    # or after we see ending punctuation followed by some tokens
    newline_count = 0
    for i, token_id in enumerate(generated_tokens):
        token_text = tokenizer.decode([token_id])

        # Check for newline
        if '\n' in token_text:
            newline_count += 1
            if newline_count >= 1:  # First newline ends the second line
                return prompt_length + i

        # Fallback: look for sentence-ending punctuation
        if i > 5 and token_text.strip() in ['.', ',', '!', '?']:
            # Check if next few tokens look like end of line
            # (This is a heuristic - poems typically end lines with punctuation)
            return prompt_length + i

    # If we didn't find a clear ending, return last generated token
    return len(tokens[0]) - 1


def extract_poem_activations(
    model: HookedTransformer,
    prompts: List[str],
    max_new_tokens: int = 30,
    device: str = "cuda",
    layers: List[int] = None
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, List[str]]:
    """
    Extract activations and targets for poem rhyme prediction.

    For each poem:
    1. Generate completion of the couplet
    2. Find the last token of the second line (the rhyme)
    3. Extract activation at the last prompt token (end of first line)
    4. Target is the last token of the second line

    Args:
        model: Language model
        prompts: List of incomplete poem prompts
        max_new_tokens: Maximum tokens to generate per prompt
        device: Device to use
        layers: Layer indices to extract (None = all layers)

    Returns:
        layer_activations: Dict mapping layer_idx -> [n_samples, d_model]
        targets: [n_samples] - token IDs of rhyming words
        generated_texts: List of full generated couplets
    """
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    layer_activations_dict = {layer_idx: [] for layer_idx in layers}
    all_targets = []
    generated_texts = []

    model.eval()
    eos_token_id = model.tokenizer.eos_token_id

    print(f"Extracting poem rhyme activations from {len(prompts)} prompts...")
    print(f"Layers: {layers}")
    print(f"Max new tokens: {max_new_tokens}\n")

    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Processing poems"):
            # Tokenize prompt
            prompt_tokens = model.to_tokens(prompt).to(device)
            prompt_length = prompt_tokens.shape[1]
            current_tokens = prompt_tokens.clone()

            # Generate until we have enough tokens for second line
            for step in range(max_new_tokens):
                logits = model(current_tokens)
                next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)
                current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0)], dim=1)

                # Check if we should stop
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break

                # Check if we've completed the second line
                # (Simple heuristic: if we've generated at least 10 tokens,
                # we probably have the second line)
                if step >= 10:
                    second_line_end = find_second_line_end(
                        current_tokens,
                        model.tokenizer,
                        prompt_length
                    )
                    # If we found a clear ending, stop
                    if second_line_end < len(current_tokens[0]) - 2:
                        break

            # Get final sequence with cache
            _, cache = model.run_with_cache(current_tokens)
            generated_text = model.tokenizer.decode(current_tokens[0])
            generated_texts.append(generated_text)

            # Find the end of the second line
            second_line_end_idx = find_second_line_end(
                current_tokens,
                model.tokenizer,
                prompt_length
            )

            # Target: last token of second line (the rhyme)
            target_token = current_tokens[0, second_line_end_idx]
            all_targets.append(target_token.cpu())

            # Extract activation at the last token of the PROMPT (end of first line)
            # This is the activation we use to predict the rhyme
            activation_position = prompt_length - 1

            for layer_idx in layers:
                layer_acts = cache["resid_post", layer_idx][0]  # [seq_len, d_model]
                act = layer_acts[activation_position, :]  # [d_model]
                layer_activations_dict[layer_idx].append(act.cpu())

    # Stack activations
    layer_activations = {}
    for layer_idx in layers:
        layer_activations[layer_idx] = torch.stack(layer_activations_dict[layer_idx])

    targets = torch.stack(all_targets)  # [n_samples]

    print(f"\n✓ Extracted {len(targets)} samples")
    print(f"  Layers: {sorted(layer_activations.keys())}")
    print(f"  Activation shape per layer: {list(layer_activations.values())[0].shape}")
    print(f"  Targets shape: {targets.shape}")

    return layer_activations, targets, generated_texts


def main():
    parser = argparse.ArgumentParser(
        description="Extract poem rhyme activations and targets"
    )

    parser.add_argument("--model_name", type=str, required=True,
                        help="Model to use")
    parser.add_argument("--poems_path", type=str, required=True,
                        help="Path to poems JSONL file")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save dataset (.pt file)")
    parser.add_argument("--max_new_tokens", type=int, default=30,
                        help="Max tokens to generate per poem")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (None = all)")

    args = parser.parse_args()

    # Parse layers
    if args.layers is not None:
        layers = [int(x.strip()) for x in args.layers.split(',')]
    else:
        layers = None

    print("=" * 80)
    print("POEM RHYME ACTIVATION EXTRACTION")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Poems: {args.poems_path}")
    print(f"Output: {args.output_path}")
    print(f"Device: {args.device}")
    print("=" * 80 + "\n")

    # Load model
    print("Loading model...")
    model = HookedTransformer.from_pretrained(args.model_name, device=args.device)
    print(f"✓ Loaded {args.model_name}")
    print(f"  Layers: {model.cfg.n_layers}")
    print(f"  d_model: {model.cfg.d_model}")
    print(f"  Vocab size: {model.cfg.d_vocab}\n")

    # Load poems
    print("Loading poems...")
    prompts = load_poem_prompts(args.poems_path)
    print(f"✓ Loaded {len(prompts)} poems\n")

    # Extract activations
    layer_activations, targets, generated_texts = extract_poem_activations(
        model=model,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        layers=layers
    )

    # Show examples
    print("\nExample generations (first 3):")
    for i, text in enumerate(generated_texts[:3]):
        target_token_text = model.tokenizer.decode([targets[i]])
        print(f"\n{i+1}. {text[:150]}...")
        print(f"   Target rhyme token: '{target_token_text}'")

    # Save dataset
    metadata = {
        'model_name': args.model_name,
        'n_samples': len(targets),
        'd_model': model.cfg.d_model,
        'vocab_size': model.cfg.d_vocab,
        'layers': layers if layers is not None else list(range(model.cfg.n_layers)),
        'task': 'poem_rhyme_prediction',
        'description': 'Predict rhyming word from activation at end of first line',
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

    print(f"\n{'=' * 80}")
    print("DATASET SAVED")
    print("=" * 80)
    print(f"Path: {output_path}")
    print(f"Samples: {len(targets)}")
    print(f"Layers: {sorted(layer_activations.keys())}")
    print("=" * 80)


if __name__ == "__main__":
    main()
