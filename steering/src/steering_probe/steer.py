"""
Apply a steering vector during model generation via forward hooks.

Position semantics:
  rel_pos <= 0  : inject into the prompt's residual stream at abs_pos = newline_pos + rel_pos
  rel_pos >  0  : inject at the rel_pos-th generated token (1-indexed)
                  using the vector computed at a prompt reference position
"""
import torch
from typing import Optional
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .extract import find_last_newline_pos, get_newline_token_id


def _resolve_device(device: str, model: PreTrainedModel) -> torch.device:
    """Resolve 'auto' to the device of the model's first parameter."""
    if device == "auto":
        return next(model.parameters()).device
    return torch.device(device)


def generate_baseline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int,
    device: str,
) -> str:
    dev = _resolve_device(device, model)
    inputs = tokenizer(prompt, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def generate_with_steering(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    vector: torch.Tensor,       # (hidden_dim,) — already on CPU
    layer: int,
    rel_pos: int,               # where to inject (see module docstring)
    alpha: float,
    max_new_tokens: int,
    device: str,
    newline_pos: Optional[int] = None,  # required when rel_pos <= 0
    normalize: bool = False,
) -> str:
    """
    Generate the couplet completion with a steering vector injected at (layer, rel_pos).

    For rel_pos <= 0:  vector is added to the prompt's residual stream at the specified
                       position during the initial prompt forward pass.
    For rel_pos >  0:  vector is added at the rel_pos-th generation step (1-indexed).
    """
    dev = _resolve_device(device, model)
    inputs = tokenizer(prompt, return_tensors="pt").to(dev)
    input_ids = inputs["input_ids"]

    if rel_pos <= 0:
        assert newline_pos is not None, "newline_pos required for prompt-position steering"
        abs_steer_pos = newline_pos + rel_pos

    # Place vec on the target layer's device (may differ from input device when sharded)
    layer_device = next(model.model.layers[layer].parameters()).device
    vec = vector.to(device=layer_device, dtype=model.dtype)
    if normalize:
        vec = vec / vec.norm().clamp(min=1e-8)

    # --- generation-step counter (shared mutable state for hooks) ---
    # gen_step[0] == 0 before first generated token; incremented by pre-hook.
    gen_step = [0]
    in_generation = [False]

    def _pre_hook(module, inp):
        hs = inp[0] if isinstance(inp, tuple) else inp
        if hs.shape[1] == 1:           # single-token → generation mode
            if in_generation[0]:
                gen_step[0] += 1
            else:
                in_generation[0] = True
                gen_step[0] = 1        # first generated token = step 1

    def _steer_hook(module, inp, output):
        hs = output[0] if isinstance(output, tuple) else output
        inject = False

        if rel_pos <= 0:
            # Prompt pass: seq_len > 1
            if hs.shape[1] > 1 and 0 <= abs_steer_pos < hs.shape[1]:
                inject = True
                pos_in_hs = abs_steer_pos
        else:
            # Generation pass: seq_len == 1, check step counter
            if hs.shape[1] == 1 and gen_step[0] == rel_pos:
                inject = True
                pos_in_hs = 0

        if inject:
            hs = hs.clone()
            hs[0, pos_in_hs] = hs[0, pos_in_hs] + alpha * vec
            if isinstance(output, tuple):
                return (hs,) + output[1:]
            return hs
        return output

    # Register on layer 0 (pre-hook for step tracking) and target layer (steer hook)
    pre_h = model.model.layers[0].register_forward_pre_hook(_pre_hook)
    steer_h = model.model.layers[layer].register_forward_hook(_steer_hook)

    try:
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        pre_h.remove()
        steer_h.remove()

    new_ids = out[0, input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)
