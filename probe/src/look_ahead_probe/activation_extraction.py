"""Activation extraction from language models using HuggingFace Transformers."""

from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def generate_and_extract_all_layers(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_k: int,
    max_new_tokens: int = 50,
    device: str = "cuda",
    layers: Optional[List[int]] = None,
    chunk_size: int = 128,
) -> Tuple[dict, torch.Tensor, List[str]]:
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

    Returns:
        layer_activations: Dict layer_idx -> Tensor[n_samples, d_model]
        targets: Tensor[n_samples, max_k]
        generated_texts: List of generated strings
    """
    n_layers = model.config.num_hidden_layers
    if layers is None:
        layers = list(range(n_layers))

    layer_act_chunks = {layer_idx: [] for layer_idx in layers}
    layer_act_buf = {layer_idx: [] for layer_idx in layers}
    all_targets = []
    generated_texts = []

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

            # Consolidate every chunk_size prompts to keep CPU RAM bounded
            if (prompt_idx + 1) % chunk_size == 0:
                for layer_idx in layers:
                    if layer_act_buf[layer_idx]:
                        layer_act_chunks[layer_idx].append(
                            torch.stack(layer_act_buf[layer_idx])
                        )
                        layer_act_buf[layer_idx] = []

    # Final consolidation
    for layer_idx in layers:
        if layer_act_buf[layer_idx]:
            layer_act_chunks[layer_idx].append(torch.stack(layer_act_buf[layer_idx]))

    layer_activations = {
        layer_idx: torch.cat(layer_act_chunks[layer_idx])
        for layer_idx in layers
    }
    targets = torch.stack(all_targets)

    return layer_activations, targets, generated_texts
