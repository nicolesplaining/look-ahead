"""Activation extraction from language models using HuggingFace Transformers."""

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def _text_cfg(model):
    """Return the text config for any supported architecture."""
    cfg = model.config
    if hasattr(cfg, "text_config"):  # Gemma3ForConditionalGeneration
        return cfg.text_config
    return cfg


def get_n_layers(model) -> int:
    cfg = _text_cfg(model)
    if hasattr(cfg, "num_hidden_layers"):
        return cfg.num_hidden_layers
    # Fallback: count actual layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "model") and hasattr(model.model, "text_model"):
        return len(model.model.text_model.layers)
    raise AttributeError(f"Cannot determine n_layers for {type(model).__name__}")


def get_hidden_size(model) -> int:
    cfg = _text_cfg(model)
    if hasattr(cfg, "hidden_size"):
        return cfg.hidden_size
    raise AttributeError(f"Cannot determine hidden_size for {type(model).__name__}")


def get_vocab_size(model) -> int:
    cfg = _text_cfg(model)
    if hasattr(cfg, "vocab_size"):
        return cfg.vocab_size
    raise AttributeError(f"Cannot determine vocab_size for {type(model).__name__}")


def generate_and_extract_all_layers(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_k: int,
    max_new_tokens: int = 50,
    device: str = "cuda",
    layers: Optional[List[int]] = None,
    chunk_size: int = 128,
    chunk_dir: Optional[str] = None,
) -> Tuple[Optional[dict], torch.Tensor, List[str], List[List[int]]]:
    """
    Generate text and extract residual-stream activations from all layers.

    For each prompt:
    1. Generate continuation with model.generate() (greedy, uses KV cache).
    2. Run a single forward pass with output_hidden_states=True on the full sequence.
    3. Collect (activation, targets) pairs for every position with max_k valid targets.

    hidden_states[0] = embedding output.
    hidden_states[L+1] = residual stream after transformer block L (= resid_post for layer L).

    Args:
        model: HuggingFace causal LM (should already be on the target device)
        tokenizer: Corresponding tokenizer
        prompts: Text prompts
        max_k: Maximum lookahead distance
        max_new_tokens: Tokens to generate per prompt
        device: Device for input tensors
        layers: Layer indices to extract (None = all)
        chunk_size: Consolidate every this many prompts to bound CPU RAM
        chunk_dir: If given, write activation chunks to this directory instead of
                   accumulating in RAM. Call merge_layer_chunks() afterwards.
                   Returns None for layer_activations when used.

    Returns:
        layer_activations: Dict layer_idx -> Tensor[n_samples, d_model], or None if chunk_dir used
        targets: Tensor[n_samples, max_k]
        generated_texts: List of generated strings
        generated_token_ids: List[List[int]] raw token IDs (no decode/re-encode roundtrip)
    """
    n_layers = get_n_layers(model)
    if layers is None:
        layers = list(range(n_layers))

    streaming = chunk_dir is not None
    if streaming:
        chunk_dir_path = Path(chunk_dir)
        chunk_dir_path.mkdir(parents=True, exist_ok=True)

    layer_act_chunks = {layer_idx: [] for layer_idx in layers}  # used when not streaming
    layer_act_buf = {layer_idx: [] for layer_idx in layers}
    all_targets = []
    generated_texts = []
    generated_token_ids = []
    chunk_idx = 0

    model.eval()
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    with torch.no_grad():
        for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Extracting activations")):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            # Step 1: generate continuation (efficient, KV cache used internally)
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )  # [1, prompt_len + generated_len]

            # Step 2: single forward pass to get hidden states at all layers
            outputs = model(generated_ids, output_hidden_states=True)
            # outputs.hidden_states: tuple of (n_layers+1) tensors, each [1, seq_len, d_model]
            # index 0 = embedding output; index L+1 = after transformer block L

            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            generated_texts.append(generated_text)
            generated_token_ids.append(generated_ids[0].tolist())

            total_len = generated_ids.shape[1]

            # Collect (activation, targets) at every position with max_k valid look-ahead tokens
            for i in range(total_len - max_k):
                targets_for_position = [
                    generated_ids[0, i + k].cpu()
                    for k in range(1, max_k + 1)
                ]
                for layer_idx in layers:
                    act = outputs.hidden_states[layer_idx + 1][0, i, :]
                    layer_act_buf[layer_idx].append(act.cpu())
                all_targets.append(torch.stack(targets_for_position))

            del outputs
            torch.cuda.empty_cache()

            # Consolidate every chunk_size prompts
            if (prompt_idx + 1) % chunk_size == 0:
                for layer_idx in layers:
                    if layer_act_buf[layer_idx]:
                        chunk_tensor = torch.stack(layer_act_buf[layer_idx])
                        if streaming:
                            torch.save(chunk_tensor,
                                       chunk_dir_path / f'layer_{layer_idx}_chunk_{chunk_idx:05d}.pt')
                        else:
                            layer_act_chunks[layer_idx].append(chunk_tensor)
                        layer_act_buf[layer_idx] = []
                chunk_idx += 1

    # Final consolidation of any remaining buffer
    for layer_idx in layers:
        if layer_act_buf[layer_idx]:
            chunk_tensor = torch.stack(layer_act_buf[layer_idx])
            if streaming:
                torch.save(chunk_tensor,
                           chunk_dir_path / f'layer_{layer_idx}_chunk_{chunk_idx:05d}.pt')
            else:
                layer_act_chunks[layer_idx].append(chunk_tensor)

    targets = torch.stack(all_targets)

    if streaming:
        return None, targets, generated_texts, generated_token_ids

    layer_activations = {
        layer_idx: torch.cat(layer_act_chunks[layer_idx])
        for layer_idx in layers
    }
    return layer_activations, targets, generated_texts, generated_token_ids


def merge_layer_chunks(chunk_dir: str, output_dir: str, layers: List[int]) -> None:
    """
    Merge per-layer chunk files written by generate_and_extract_all_layers into
    final layer_N.pt files. Processes one layer at a time to keep peak RAM low.
    Deletes chunk files after merging each layer.
    """
    chunk_dir_path = Path(chunk_dir)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    for layer_idx in layers:
        chunk_files = sorted(chunk_dir_path.glob(f'layer_{layer_idx}_chunk_*.pt'))
        if not chunk_files:
            print(f"  WARNING: no chunks found for layer {layer_idx}")
            continue
        chunks = [torch.load(f, weights_only=False) for f in chunk_files]
        merged = torch.cat(chunks)
        del chunks
        torch.save(merged, output_dir_path / f'layer_{layer_idx}.pt')
        n_samples = merged.shape[0]
        del merged
        for f in chunk_files:
            f.unlink()
        print(f"  layer {layer_idx}: {n_samples} samples")

    # Remove chunk dir if now empty
    try:
        chunk_dir_path.rmdir()
    except OSError:
        pass
