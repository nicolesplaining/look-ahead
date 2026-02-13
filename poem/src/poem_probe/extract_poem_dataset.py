#!/usr/bin/env python3
"""
Extract activations and targets for poem rhyme prediction.

For each poem prompt (incomplete couplet), generates the second line and creates:
- Input:  activation at the last token of the FIRST line (end of prompt)
- Target: the last token of the second line BEFORE the terminating newline
          (i.e. the rhyming word)

Generation stops as soon as a newline token is emitted.  Examples where no
newline is generated within max_new_tokens are skipped.

The activation at the last prompt token is independent of the generated
continuation (due to causal masking), so we forward-pass only the prompt
for activations and use the generated output only to find the target token.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 16,
    device: str = "cuda",
    layers: Optional[List[int]] = None,
    chunk_size: int = 128,
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, List[str]]:
    """
    Extract activations and targets for poem rhyme prediction.

    For each poem:
    1. Generate tokens until a newline is produced (or max_new_tokens).
    2. Skip the example if no newline is generated.
    3. Target = the token immediately before the newline (the rhyming word).
    4. Activation = residual stream at the last prompt token.
       (Causal masking ensures this is independent of the generated continuation,
       so we forward-pass only the prompt for activations.)

    Args:
        model: HuggingFace causal LM (should already be on the target device)
        tokenizer: Corresponding tokenizer
        prompts: Incomplete poem prompts (first line already included)
        max_new_tokens: Maximum tokens to generate before giving up
        device: Device for input tensors
        layers: Layer indices to extract (None = all layers)
        chunk_size: Consolidate every this many kept examples to bound CPU RAM

    Returns:
        layer_activations: Dict layer_idx -> [n_kept, d_model]
        targets:           [n_kept] token IDs of the rhyming words
        generated_texts:   List of full generated couplet strings
    """
    if layers is None:
        layers = list(range(model.config.num_hidden_layers))

    layer_act_chunks = {layer_idx: [] for layer_idx in layers}
    layer_act_buf = {layer_idx: [] for layer_idx in layers}
    all_targets: List[torch.Tensor] = []
    generated_texts: List[str] = []

    model.eval()
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    n_skipped = 0
    kept_count = 0

    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Extracting poem activations"):
            prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            prompt_length = prompt_tokens.shape[1]

            # Generate up to max_new_tokens; find the first newline in the output
            generated_ids = model.generate(
                prompt_tokens,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )  # [1, prompt_len + generated_len]

            # Scan generated portion for first newline token
            newline_pos: Optional[int] = None
            for i in range(prompt_length, generated_ids.shape[1]):
                tok_text = tokenizer.decode([generated_ids[0, i].item()])
                if '\n' in tok_text:
                    newline_pos = i
                    break

            # Skip if no newline was generated
            if newline_pos is None:
                n_skipped += 1
                continue

            # Target: last token before the newline (the rhyming word)
            target_idx = newline_pos - 1
            if target_idx < prompt_length:
                # Newline was the very first generated token — nothing to probe
                n_skipped += 1
                continue

            # Activation at last prompt token via forward pass on prompt only.
            # Causal masking guarantees this is identical to what we'd get from
            # the full sequence at the same position.
            outputs = model(prompt_tokens, output_hidden_states=True)
            # hidden_states[0] = embedding; hidden_states[L+1] = after block L

            activation_position = prompt_length - 1
            for layer_idx in layers:
                act = outputs.hidden_states[layer_idx + 1][0, activation_position, :]
                layer_act_buf[layer_idx].append(act.cpu())

            del outputs
            torch.cuda.empty_cache()

            target_token = generated_ids[0, target_idx]
            all_targets.append(target_token.cpu())
            generated_texts.append(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

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
            layer_act_chunks[layer_idx].append(torch.stack(layer_act_buf[layer_idx]))

    if n_skipped:
        print(f"Skipped {n_skipped} examples (no newline generated within {max_new_tokens} tokens)")

    layer_activations = {
        layer_idx: torch.cat(layer_act_chunks[layer_idx])
        for layer_idx in layers
    }
    targets = torch.stack(all_targets)

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
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"✓ Loaded {args.model_name}  "
          f"(layers={model.config.num_hidden_layers}, d_model={model.config.hidden_size})\n")

    print("Loading poems...")
    prompts = load_poem_prompts(args.poems_path, max_prompts=args.max_prompts)
    print(f"✓ Loaded {len(prompts)} poem prompts\n")

    layer_activations, targets, generated_texts = extract_poem_activations(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        layers=layers,
    )

    print("\nExample generations (first 3):")
    for i, text in enumerate(generated_texts[:3]):
        target_text = tokenizer.decode([targets[i].item()])
        print(f"\n{i+1}. {text[:150]}")
        print(f"   Rhyme target token: {repr(target_text)}")

    metadata = {
        'model_name': args.model_name,
        'n_samples': len(targets),
        'd_model': model.config.hidden_size,
        'vocab_size': model.config.vocab_size,
        'layers': layers if layers is not None else list(range(model.config.num_hidden_layers)),
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
