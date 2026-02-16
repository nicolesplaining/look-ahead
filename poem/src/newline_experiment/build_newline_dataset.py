#!/usr/bin/env python3
"""
Build newline_experiment dataset from existing activations .pt files.

Extracts i=0 (newline token) activations and computes targets for
k = 0 .. max_k, where k=0 is the rhyme word and k>0 goes backwards
through the second line of the couplet.

Input:  activations_{train,val}.pt   (existing poem extraction format)
Output: newline_{train,val}.pt

Output schema:
    layer_activations : Dict[layer_idx -> Tensor[M, d_model]]
    targets           : Tensor[M, max_k+1]   # -1 where unavailable
    poem_texts        : List[str]             # one per poem (M entries)
    metadata          : dict

Usage:
    python -m newline_experiment.build_newline_dataset \\
        --input  poem/data/activations_train.pt \\
        --output poem/data/newline_train.pt \\
        --model_name Qwen/Qwen2.5-7B \\
        --max_k 5
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoTokenizer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Target recovery
# ---------------------------------------------------------------------------

def _get_second_line_targets(
    text: str,
    tokenizer,
    max_k: int,
    expected_k0: Optional[int] = None,
) -> List[Optional[int]]:
    """
    Tokenise `text` (a full "A rhyming couplet:\\nLine1\\nLine2\\n" string) and
    return [target_k0, target_k1, ..., target_kmax_k].

    target_ki = token at k steps before the final token of the second line.
    k=0 is the rhyme word (last token before the terminating \\n of line 2).

    Returns None entries for k values where the second line is too short.
    If expected_k0 is given and doesn't match, all entries are None (mismatch
    means the re-tokenisation differs from the original extraction).
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    # Scan for positions whose decoded token contains '\n'
    nl_positions = [
        i for i, tid in enumerate(token_ids)
        if '\n' in tokenizer.decode([tid])
    ]

    # Need at least two newlines: end of first line + end of second line
    if len(nl_positions) < 2:
        return [None] * (max_k + 1)

    last_nl   = nl_positions[-1]   # end of second line
    second_nl = nl_positions[-2]   # end of first line

    # Targets are at positions last_nl-1 (k=0), last_nl-2 (k=1), ...
    # but must stay within the second line (> second_nl)
    targets = []
    for k in range(max_k + 1):
        pos = last_nl - 1 - k
        if pos <= second_nl:
            targets.append(None)
        else:
            targets.append(int(token_ids[pos]))

    # Verify k=0 against the stored target if provided
    if expected_k0 is not None and targets[0] is not None:
        if targets[0] != expected_k0:
            return [None] * (max_k + 1)

    return targets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_newline_dataset(
    input_path: str,
    output_path: str,
    model_name: str,
    max_k: int = 5,
) -> None:
    print(f"Loading: {input_path}")
    data = torch.load(input_path, weights_only=False)

    layer_acts_all = data['layer_activations']   # {layer_idx: Tensor[N, d]}
    targets_all    = data['targets']             # Tensor[N]
    i_values       = data['i_values']            # Tensor[N]
    generated_texts: List[str] = data['generated_texts']
    metadata = data['metadata']

    # -----------------------------------------------------------------------
    # Filter to i=0 samples
    # -----------------------------------------------------------------------
    i0_mask = (i_values == 0)
    n_i0 = int(i0_mask.sum())
    print(f"Total samples: {len(i_values)}  |  i=0 samples: {n_i0}  |  poems: {len(generated_texts)}")

    if n_i0 != len(generated_texts):
        raise ValueError(
            f"Expected one i=0 sample per poem, but got {n_i0} i=0 samples "
            f"and {len(generated_texts)} poems. The .pt file may be malformed."
        )

    i0_layer_acts = {
        layer_idx: acts[i0_mask].clone()
        for layer_idx, acts in layer_acts_all.items()
    }
    i0_targets_k0 = targets_all[i0_mask]  # Tensor[M], stored rhyme-word tokens

    # -----------------------------------------------------------------------
    # Load tokenizer
    # -----------------------------------------------------------------------
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # -----------------------------------------------------------------------
    # Recover targets for k = 0 .. max_k via re-tokenisation
    # -----------------------------------------------------------------------
    print(f"\nRecovering targets for k=0..{max_k} from generated texts...")
    all_targets: List[List[int]] = []   # shape [M, max_k+1], -1 = unavailable
    n_mismatch = 0
    n_short    = 0

    for poem_idx, text in enumerate(tqdm(generated_texts, desc="Building targets")):
        expected_k0 = int(i0_targets_k0[poem_idx].item())
        row = _get_second_line_targets(text, tokenizer, max_k, expected_k0=expected_k0)

        if row[0] is None:
            n_mismatch += 1

        # Replace None with -1 for storage
        all_targets.append([t if t is not None else -1 for t in row])

        # Count poems where any k > 0 is unavailable
        if any(t == -1 for t in all_targets[-1][1:]):
            n_short += 1

    targets_tensor = torch.tensor(all_targets, dtype=torch.long)  # [M, max_k+1]

    print(f"\n✓ Built targets tensor: {list(targets_tensor.shape)}")
    print(f"  Tokenisation mismatches (k=0 mismatch, all set to -1): {n_mismatch}")
    print(f"  Poems with short second lines (some k > 0 unavailable): {n_short}")

    for k in range(max_k + 1):
        valid = int((targets_tensor[:, k] != -1).sum())
        print(f"  k={k}: {valid}/{len(generated_texts)} valid samples")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    out_metadata = {
        **metadata,
        'max_k':       max_k,
        'task':        'newline_experiment',
        'source_file': str(input_path),
    }

    dataset = {
        'layer_activations': i0_layer_acts,
        'targets':           targets_tensor,
        'poem_texts':        generated_texts,
        'metadata':          out_metadata,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)
    print(f"\nSaved: {output_path}")
    print(f"  Poems: {len(generated_texts)}")
    print(f"  Layers: {sorted(i0_layer_acts.keys())}")
    print(f"  Activation shape per layer: {list(next(iter(i0_layer_acts.values())).shape)}")


def main():
    parser = argparse.ArgumentParser(
        description="Build newline_experiment dataset from existing activations .pt"
    )
    parser.add_argument("--input",      type=str, required=True,
                        help="Path to existing activations .pt (train or val)")
    parser.add_argument("--output",     type=str, required=True,
                        help="Output .pt path")
    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFace model name (for tokenizer)")
    parser.add_argument("--max_k",      type=int, default=5,
                        help="Max k to extract (default: 5)")
    args = parser.parse_args()

    build_newline_dataset(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model_name,
        max_k=args.max_k,
    )


if __name__ == "__main__":
    main()
