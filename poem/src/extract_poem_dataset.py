#!/usr/bin/env python3
"""
Extract activations for poem rhyme prediction with i-indexed positions.

Prompt format:
    "A rhyming couplet:\n{First Line}\n"

Position indexing (i):
    i = 0   : the final \\n at the end of the first line (last prompt token)
    i < 0   : tokens before the \\n, going back into the first line
    i > 0   : tokens in the generated second line after the \\n
    target  : last token before the terminating \\n of the second line

For each poem:
1. Generate the second line.
2. Find target = last token before the terminating \\n.
3. Store activations for i = max(-max_back, -(available tokens)) up to i = target_i (inclusive).
4. Single forward pass on the full generated sequence.

Dataset schema (.pt):
    layer_activations: Dict[layer_idx -> Tensor[N, d_model]]   # bfloat16
    targets:           Tensor[N]      # rhyming word token ID, one per sample
    i_values:          Tensor[N]      # position index relative to first-line \\n
    generated_texts:   List[str]      # one entry per kept poem
    metadata:          dict
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def _text_cfg(model):
    """Return the text config for any supported architecture."""
    cfg = model.config
    if hasattr(cfg, "text_config"):  # Gemma3ForConditionalGeneration
        return cfg.text_config
    return cfg


def _get_model_layers(model):
    """Return the transformer decoder layer list for any supported architecture."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "text_model") \
            and hasattr(model.model.text_model, "layers"):
        return model.model.text_model.layers
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
        if hasattr(lm, "layers"):
            return lm.layers
    raise AttributeError(f"Cannot find transformer layers for {type(model).__name__}")


def get_n_layers(model) -> int:
    cfg = _text_cfg(model)
    if hasattr(cfg, "num_hidden_layers"):
        return cfg.num_hidden_layers
    return len(_get_model_layers(model))


def get_hidden_size(model) -> int:
    cfg = _text_cfg(model)
    if hasattr(cfg, "hidden_size"):
        return cfg.hidden_size
    return next(iter(_get_model_layers(model)[0].parameters())).shape[-1]


def get_vocab_size(model) -> int:
    cfg = _text_cfg(model)
    if hasattr(cfg, "vocab_size"):
        return cfg.vocab_size
    raise AttributeError(f"Cannot find vocab_size for {type(model).__name__}")


def load_poem_prompts(jsonl_path: str, max_prompts: Optional[int] = None) -> List[str]:
    prompts = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            prompts.append(data['text'])
            if max_prompts is not None and len(prompts) >= max_prompts:
                break
    return prompts


def _try_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_tokens: torch.Tensor,
    prompt_length: int,
    max_new_tokens: int,
    pad_token_id: int,
    do_sample: bool = False,
) -> Optional[Tuple[torch.Tensor, int]]:
    """
    Attempt to generate a second line.

    Returns (generated_ids, target_seq_pos) on success, or None if:
      - no newline was generated within max_new_tokens
      - the second line is empty (newline was the very first generated token)
    """
    generated_ids = model.generate(
        prompt_tokens,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=0.7 if do_sample else None,
        pad_token_id=pad_token_id,
    )

    newline2_pos: Optional[int] = None
    for i in range(prompt_length, generated_ids.shape[1]):
        if '\n' in tokenizer.decode([generated_ids[0, i].item()]):
            newline2_pos = i
            break

    if newline2_pos is None or newline2_pos == prompt_length:
        return None

    # Scan backwards past punctuation to find the last alphabetic token
    target_seq_pos = newline2_pos - 1
    while target_seq_pos > prompt_length:
        decoded = tokenizer.decode([generated_ids[0, target_seq_pos].item()])
        if any(c.isalpha() for c in decoded):
            break
        target_seq_pos -= 1
    else:
        return None  # second line has no alphabetic token

    return generated_ids, target_seq_pos


def extract_poem_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_back: int = 8,
    max_new_tokens: int = 32,
    device: str = "cuda",
    layers: Optional[List[int]] = None,
    chunk_size: int = 128,
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor, List[str]]:
    """
    Extract activations for all positions i = -max_back ... target_i (inclusive).

    i = 0 is the final \\n at the end of the first line (last prompt token).
    Negative i are tokens earlier in the first line.
    Positive i are tokens in the generated second line.
    target_i = target_pos - newline_seq_pos (always >= 1).

    Returns:
        layer_activations: Dict[layer_idx -> Tensor[N, d_model]]
        targets:           Tensor[N]   rhyming word token ID per sample
        i_values:          Tensor[N]   position index relative to first-line \\n
        generated_texts:   List[str]   one entry per kept poem
    """
    if layers is None:
        layers = list(range(get_n_layers(model)))

    layer_act_chunks = {layer_idx: [] for layer_idx in layers}
    layer_act_buf = {layer_idx: [] for layer_idx in layers}
    all_targets: List[torch.Tensor] = []
    all_i_values: List[torch.Tensor] = []
    generated_texts: List[str] = []
    n_skipped = 0
    n_samples = 0

    model.eval()
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    with torch.no_grad():
        for poem_idx, prompt in enumerate(tqdm(prompts, desc="Extracting poem activations")):
            prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            prompt_length = prompt_tokens.shape[1]

            # i = 0 is the last prompt token (the final \n of the first line)
            newline_seq_pos = prompt_length - 1

            # Try up to 3 times: first greedy, then with sampling.
            gen_result = None
            for attempt in range(3):
                gen_result = _try_generate(
                    model, tokenizer, prompt_tokens, prompt_length,
                    max_new_tokens, pad_token_id,
                    do_sample=(attempt > 0),
                )
                if gen_result is not None:
                    break

            if gen_result is None:
                n_skipped += 1
                continue

            generated_ids, target_seq_pos = gen_result
            target_token = generated_ids[0, target_seq_pos]

            # Valid range in sequence space:
            #   start: as far back as possible, capped at max_back tokens before newline
            #          but not before position 0
            start_seq_pos = max(newline_seq_pos - max_back, 0)
            # end: target position (inclusive)
            end_seq_pos = target_seq_pos

            # Single forward pass on full generated sequence
            outputs = model(generated_ids, output_hidden_states=True)

            for seq_pos in range(start_seq_pos, end_seq_pos + 1):
                i_val = seq_pos - newline_seq_pos  # negative for first-line tokens
                for layer_idx in layers:
                    act = outputs.hidden_states[layer_idx + 1][0, seq_pos, :]
                    layer_act_buf[layer_idx].append(act.cpu())
                all_targets.append(target_token.cpu())
                all_i_values.append(torch.tensor(i_val, dtype=torch.long))

            del outputs
            torch.cuda.empty_cache()

            generated_texts.append(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
            n_samples += (end_seq_pos - start_seq_pos + 1)

            # Periodic consolidation to keep CPU RAM bounded
            if (poem_idx + 1) % chunk_size == 0:
                for layer_idx in layers:
                    if layer_act_buf[layer_idx]:
                        layer_act_chunks[layer_idx].append(torch.stack(layer_act_buf[layer_idx]))
                        layer_act_buf[layer_idx] = []

    # Final consolidation
    for layer_idx in layers:
        if layer_act_buf[layer_idx]:
            layer_act_chunks[layer_idx].append(torch.stack(layer_act_buf[layer_idx]))

    if n_skipped:
        print(f"Skipped {n_skipped} poems (no newline generated or second line too short)")

    layer_activations = {
        layer_idx: torch.cat(layer_act_chunks[layer_idx])
        for layer_idx in layers
    }
    targets = torch.stack(all_targets)
    i_values = torch.stack(all_i_values)

    i_min, i_max = int(i_values.min()), int(i_values.max())
    print(f"\n✓ Extracted {n_samples} samples from {len(generated_texts)} poems")
    print(f"  i range: {i_min} to {i_max}")
    print(f"  Activation shape per layer: {list(layer_activations.values())[0].shape}")

    return layer_activations, targets, i_values, generated_texts


def main():
    parser = argparse.ArgumentParser(
        description="Extract poem rhyme activations with i-indexed positions"
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--poems_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_back", type=int, default=8,
                        help="How many tokens before the first-line \\n to store (default: 8)")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (default: all)")

    args = parser.parse_args()

    layers = None
    if args.layers is not None:
        layers = [int(x.strip()) for x in args.layers.split(',')]

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map=args.device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"✓ {args.model_name} "
          f"(layers={get_n_layers(model)}, d_model={get_hidden_size(model)})\n")

    print("Loading poems...")
    prompts = load_poem_prompts(args.poems_path, max_prompts=args.max_prompts)
    print(f"✓ Loaded {len(prompts)} poem prompts\n")

    layer_activations, targets, i_values, generated_texts = extract_poem_activations(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_back=args.max_back,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        layers=layers,
    )

    print("\nExamples (first 3 kept poems):")
    for i, text in enumerate(generated_texts[:3]):
        print(f"  {i+1}. {text[:120]}")

    # Count samples per i value
    i_counts = {}
    for iv in i_values.tolist():
        i_counts[iv] = i_counts.get(iv, 0) + 1
    i_min, i_max = min(i_counts), max(i_counts)
    print(f"\nSamples per i: {dict(sorted(i_counts.items()))}")

    metadata = {
        'model_name': args.model_name,
        'max_back': args.max_back,
        'n_poems': len(generated_texts),
        'n_samples': len(targets),
        'd_model': get_hidden_size(model),
        'vocab_size': get_vocab_size(model),
        'layers': layers if layers is not None else list(range(get_n_layers(model))),
        'task': 'poem_rhyme_prediction_i_indexed',
        'i_range': [i_min, i_max],
    }

    dataset = {
        'layer_activations': layer_activations,
        'targets': targets,
        'i_values': i_values,
        'generated_texts': generated_texts,
        'metadata': metadata,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)

    print(f"\nDataset saved to: {output_path}")
    print(f"  Samples: {len(targets)}, Poems: {len(generated_texts)}")
    print(f"  Layers:  {sorted(layer_activations.keys())}")


if __name__ == "__main__":
    main()
