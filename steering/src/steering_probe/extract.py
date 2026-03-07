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


def get_layers(model) -> torch.nn.ModuleList:
    """Return the transformer decoder layers for any supported architecture."""
    # Standard path: Qwen, Llama, Gemma2, Gemma3ForCausalLM
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # Gemma3ForConditionalGeneration: model.model.text_model.layers
    if (hasattr(model, "model") and hasattr(model.model, "text_model")
            and hasattr(model.model.text_model, "layers")):
        return model.model.text_model.layers
    # Fallback: language_model.model.layers or language_model.layers
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
        if hasattr(lm, "layers"):
            return lm.layers
    raise AttributeError(
        f"Cannot find transformer layers for {type(model).__name__}. "
        "Tried: model.model.layers, model.model.text_model.layers, "
        "model.language_model.model.layers, model.language_model.layers."
    )


def get_hidden_size(model) -> int:
    """Return the hidden dimension for any supported architecture."""
    cfg = model.config
    if hasattr(cfg, "hidden_size"):
        return cfg.hidden_size
    # Gemma3ForConditionalGeneration: text config is nested
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        return cfg.text_config.hidden_size
    # Fallback: infer from the actual layer weights
    layers = get_layers(model)
    return next(iter(layers[0].parameters())).shape[-1]


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
    n_layers = len(get_layers(model))
    if layers is None:
        layers = list(range(n_layers))
    hidden_dim = get_hidden_size(model)
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
            hooks.append(get_layers(model)[l].register_forward_hook(_make_hook(l)))

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
