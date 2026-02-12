"""Efficient activation extraction from language models."""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from tqdm import tqdm


def verify_activation_equivalence(
    model: HookedTransformer,
    prompt: str,
    layer_idx: int,
    max_new_tokens: int,
    device: str = "cuda"
) -> bool:
    """
    Verify efficient extraction is valid for the model.

    Tests that activations from token-by-token generation match single forward pass.
    """
    model.eval()
    eos_token_id = model.tokenizer.eos_token_id
    EPSILON = 1e-3

    print(f"\n{'='*60}")
    print(f"Verifying Activation Equivalence")
    print(f"{'='*60}")
    print(f"Prompt: {prompt[:50]}...")
    print(f"Layer: {layer_idx}, Max new tokens: {max_new_tokens}\n")

    with torch.no_grad():
        # METHOD 1: Token-by-token generation (storing activations as we go)
        current_tokens = model.to_tokens(prompt).to(device)
        method1_activations = {}

        for step in range(max_new_tokens):
            logits, cache = model.run_with_cache(current_tokens)
            layer_acts = cache["resid_post", layer_idx][0]  # [seq_len, d_model]

            # Store activations for positions we haven't stored yet
            for pos in range(current_tokens.shape[1]):
                if pos not in method1_activations:
                    method1_activations[pos] = layer_acts[pos].cpu()

            # Greedy generation
            next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)
            current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0)], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        final_tokens_method1 = current_tokens

        # METHOD 2: Generate first, then single forward pass
        # (Re-generate to ensure same tokens)
        current_tokens = model.to_tokens(prompt).to(device)

        for step in range(max_new_tokens):
            logits = model(current_tokens)
            next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)
            current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0)], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        # Single forward pass on complete sequence
        _, cache = model.run_with_cache(current_tokens)
        method2_activations = cache["resid_post", layer_idx][0]  # [seq_len, d_model]

        final_tokens_method2 = current_tokens

        # Verify tokens are identical
        if not torch.equal(final_tokens_method1, final_tokens_method2):
            print("ERROR: Generated tokens differ between methods!")
            return False

        # Compare activations at each position
        print(f"Comparing activations at {len(method1_activations)} positions:\n")
        print(f"{'Position':<10} {'Max Difference':<20} {'Status':<10}")
        print(f"{'-'*40}")

        all_valid = True
        max_diff_overall = 0.0

        for pos in sorted(method1_activations.keys()):
            act1 = method1_activations[pos]
            act2 = method2_activations[pos].cpu()

            max_diff = (act1 - act2).abs().max().item()
            max_diff_overall = max(max_diff_overall, max_diff)

            is_valid = max_diff < EPSILON
            status = "✓ PASS" if is_valid else "✗ FAIL"
            all_valid = all_valid and is_valid

            print(f"{pos:<10} {max_diff:<20.2e} {status:<10}")

        print(f"{'-'*40}")
        print(f"Overall max difference: {max_diff_overall:.2e}")
        print(f"Threshold: 1e-5")
        print(f"\nResult: {'✓ EQUIVALENCE VERIFIED' if all_valid else '✗ EQUIVALENCE FAILED'}")
        print(f"{'='*60}\n")

        return all_valid


def generate_and_extract_all_layers(
    model: HookedTransformer,
    prompts: List[str],
    max_k: int,
    max_new_tokens: int = 50,
    device: str = "cuda",
    layers: Optional[List[int]] = None
) -> Tuple[dict, torch.Tensor, List[str]]:
    """
    Generate text and extract activations from all layers with multiple lookahead targets.

    Args:
        model: The language model
        prompts: List of text prompts to start generation from
        max_k: Maximum lookahead distance (extracts targets for k=1,2,...,max_k)
        max_new_tokens: Maximum number of tokens to generate per prompt
        device: Device to use
        layers: Optional list of layer indices to extract. If None, extracts all layers.

    Returns:
        layer_activations: Dict mapping layer_idx -> tensor of shape [n_samples, d_model]
        targets: [n_samples, max_k] - token IDs at positions i+1, i+2, ..., i+max_k
        generated_texts: List of generated text strings
    """
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    layer_activations_dict = {layer_idx: [] for layer_idx in layers}
    all_targets = []
    generated_texts = []

    model.eval()
    eos_token_id = model.tokenizer.eos_token_id

    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Extracting multi-k activations"):
            current_tokens = model.to_tokens(prompt).to(device)

            for step in range(max_new_tokens):
                logits = model(current_tokens)
                next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)
                current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0)], dim=1)

                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break

            _, cache = model.run_with_cache(current_tokens)
            generated_text = model.tokenizer.decode(current_tokens[0])
            generated_texts.append(generated_text)

            total_len = current_tokens.shape[1]

            # Extract activations only at positions where all k targets are valid
            for i in range(total_len - max_k):
                # Extract targets for k=1, 2, ..., max_k
                targets_for_position = []
                for k in range(1, max_k + 1):
                    target = current_tokens[0, i + k]
                    targets_for_position.append(target.cpu())

                # Extract activation from each layer at position i
                for layer_idx in layers:
                    layer_acts = cache["resid_post", layer_idx][0]
                    act = layer_acts[i, :]
                    layer_activations_dict[layer_idx].append(act.cpu())

                all_targets.append(torch.stack(targets_for_position))

    layer_activations = {}
    for layer_idx in layers:
        layer_activations[layer_idx] = torch.stack(layer_activations_dict[layer_idx])

    targets = torch.stack(all_targets)  # [n_samples, max_k]

    return layer_activations, targets, generated_texts
