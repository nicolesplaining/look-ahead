"""
Activation extraction for steering vector computation.

Position convention:
  i = 0  : last newline token of the first couplet line
  i < 0  : tokens before the newline  (i = -1 is one before, etc.)
  (generation positions i > 0 are handled in steer.py)
"""
import torch
from typing import Dict, List, Optional
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def get_newline_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    ids = tokenizer.encode("\n", add_special_tokens=False)
    return ids[-1]


def find_last_newline_pos(input_ids: torch.Tensor, newline_id: int) -> int:
    """Return absolute index of the last newline token (1-D tensor)."""
    positions = (input_ids == newline_id).nonzero(as_tuple=True)[0]
    if len(positions) == 0:
        return int(input_ids.shape[0]) - 1
    return int(positions[-1])


def extract_scheme_means(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: List[dict],       # [{'id', 'scheme', 'text'}, ...]
    context_window: int,        # how many tokens before newline to include
    layers: Optional[List[int]],
    device: str,
) -> Dict[int, Dict[int, Dict[int, torch.Tensor]]]:
    """
    Run a forward pass for each example, capture residual-stream activations
    at each (layer, relative_position), and return per-scheme means.

    Returns:
        {scheme: {layer: {rel_pos: mean_tensor(hidden_dim)}}}
    where rel_pos in [-(context_window), ..., 0].
    """
    n_layers = len(model.model.layers)
    if layers is None:
        layers = list(range(n_layers))
    hidden_dim = model.config.hidden_size
    newline_id = get_newline_token_id(tokenizer)

    # Accumulators: scheme -> layer -> rel_pos -> (sum, count)
    sums: Dict[int, Dict[int, Dict[int, torch.Tensor]]] = {}
    counts: Dict[int, Dict[int, Dict[int, int]]] = {}

    for ex in examples:
        scheme = int(ex["scheme"])
        if scheme not in sums:
            sums[scheme] = {l: {} for l in layers}
            counts[scheme] = {l: {} for l in layers}

        inputs = tokenizer(ex["text"], return_tensors="pt").to(device)
        input_ids = inputs["input_ids"][0]
        newline_pos = find_last_newline_pos(input_ids, newline_id)

        captured: Dict[int, torch.Tensor] = {}
        hooks = []

        for l in layers:
            def _make_hook(layer_idx: int):
                def _hook(module, inp, output):
                    hs = output[0] if isinstance(output, tuple) else output
                    captured[layer_idx] = hs[0].detach().float().cpu()  # (seq_len, d)
                return _hook
            hooks.append(model.model.layers[l].register_forward_hook(_make_hook(l)))

        with torch.no_grad():
            model(**inputs)

        for h in hooks:
            h.remove()

        for rel_pos in range(-context_window, 1):  # -context_window ... 0
            abs_pos = newline_pos + rel_pos
            if abs_pos < 0 or abs_pos >= int(input_ids.shape[0]):
                continue
            for l in layers:
                if l not in captured:
                    continue
                act = captured[l][abs_pos]  # (hidden_dim,)
                if rel_pos not in sums[scheme][l]:
                    sums[scheme][l][rel_pos] = torch.zeros(hidden_dim)
                    counts[scheme][l][rel_pos] = 0
                sums[scheme][l][rel_pos] += act
                counts[scheme][l][rel_pos] += 1

    # Compute means
    means: Dict[int, Dict[int, Dict[int, torch.Tensor]]] = {}
    for scheme in sums:
        means[scheme] = {}
        for l in layers:
            means[scheme][l] = {}
            for rel_pos in sums[scheme][l]:
                c = counts[scheme][l][rel_pos]
                if c > 0:
                    means[scheme][l][rel_pos] = sums[scheme][l][rel_pos] / c
    return means
