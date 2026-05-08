"""
Causal Head Gating (CHG) for Gemma-3-27B-IT on the rhyming-couplet task.

Implements the algorithm from Nam et al. 2025 (arXiv:2505.13737):
  - Multiplicative gates G in [0,1]^{L x H} on each attention head's A·V
    output, applied just before the output projection W^O
  - Loss: NLL on next-token prediction + lambda * sum sigmoid^{-1}(G)
  - Two fits: lambda > 0 (G+, retention pressure) and lambda < 0
    (G-, removal pressure), shared init from a lambda=0 warmup
  - Taxonomy: facilitating (high G+, high G-), interfering (low G+,
    low G-), irrelevant (high G+, low G-)

This script is a SMOKE TEST: 5 prompt pairs from our existing patching
experiment, ~50 training steps, to verify the gating mechanism works
end-to-end on Gemma-3-27B before scaling up.
"""

import gc
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "google/gemma-3-27b-it"
RUN_NAME   = "gemma3_27b_chg_smoke"

# Hand-curated 5 pair dataset from our existing patching runs.
# Each entry: prompt up through "they" / "until they all" / etc., then
# a rhyme-correct continuation. The model already produces these greedily;
# we use them as supervised targets for CHG fitting.
DATA = [
    {
        "pair_id": "doom_dread",
        "prompt": "A rhyming couplet:\nThe empty house was filled with silent doom,\nwhen suddenly they",
        "target": " heard a ghostly boom.",
    },
    {
        "pair_id": "bliss_joy",
        "prompt": "A rhyming couplet:\nThe children laughed in bliss,\nuntil they all",
        "target": " shared a sweet kiss.",
    },
    {
        "pair_id": "dark_night",
        "prompt": "A rhyming couplet:\nShe wandered home alone into the dark,\nand then she",
        "target": " heard the dog start to bark.",
    },
    {
        "pair_id": "grief_pain",
        "prompt": "A rhyming couplet:\nI never knew the depth of such grief,\nas though the",
        "target": " world had lost all relief.",
    },
    {
        "pair_id": "fright_fear",
        "prompt": "A rhyming couplet:\nShe felt a sudden sense of fright,\nand hoped that",
        "target": " day would soon be bright.",
    },
]

# Smoke-test hyperparameters (retune: weaker reg + more steps)
N_STEPS_WARMUP = 30
N_STEPS_MAIN   = 100
LR             = 0.05
LAMBDA_POS     = 0.01
LAMBDA_NEG     = -0.01
SEED           = 0

# ── Model ─────────────────────────────────────────────────────────────────────

def load_model():
    print(f"Loading {MODEL_NAME} (bf16, no quantization)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Gemma-3-27B in 4-bit produces NaN logits (bnb 4-bit incompatibility);
    # load in bf16 instead. 27B × 2 bytes = ~54 GB, fits in 96 GB GPU.
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    # Don't .eval() — we need backward through frozen 4-bit weights.
    # Setting requires_grad=False on all weights:
    for p in model.parameters():
        p.requires_grad_(False)

    cfg = model.config.text_config
    n_layers = cfg.num_hidden_layers
    n_heads  = cfg.num_attention_heads
    d_head   = cfg.head_dim if hasattr(cfg, "head_dim") else cfg.hidden_size // n_heads
    print(f"Loaded. Layers={n_layers} Heads/layer={n_heads} d_head={d_head}", flush=True)
    return model, tokenizer, n_layers, n_heads, d_head


def get_layers(model):
    return model.model.language_model.layers


def get_input_device(model):
    return model.model.language_model.embed_tokens.weight.device


# ── Gates ──────────────────────────────────────────────────────────────────────

class GateBank:
    """L x H gate logits + hooks on each layer's self_attn.o_proj input."""

    def __init__(self, model, n_layers, n_heads, d_head, init_logits=None):
        self.n_layers = n_layers
        self.n_heads  = n_heads
        self.d_head   = d_head
        device = get_input_device(model)
        if init_logits is None:
            # Init near-identity (sigmoid(4)~=0.98) so initial forward pass is
            # close to unperturbed model. Halving every head from logit=0 init
            # destabilizes the 62-layer Gemma to NaN.
            init = 4.0 * torch.ones(n_layers, n_heads, device=device, dtype=torch.float32)
        else:
            init = init_logits.clone().to(device)
        self.G_logits = nn.Parameter(init)
        self.handles = []
        self._install(model)

    def _install(self, model):
        layers = get_layers(model)
        for i, layer in enumerate(layers):
            attn = layer.self_attn
            o_proj = attn.o_proj
            h = o_proj.register_forward_pre_hook(self._make_hook(i))
            self.handles.append(h)

    def _make_hook(self, layer_idx):
        n_heads = self.n_heads
        d_head  = self.d_head
        def hook(module, args):
            x = args[0]  # [B, T, n_heads * d_head] for standard Gemma attn
            B, T, D = x.shape
            assert D == n_heads * d_head, \
                f"Unexpected o_proj input shape {x.shape}, expected last={n_heads*d_head}"
            xv = x.view(B, T, n_heads, d_head)
            gates = torch.sigmoid(self.G_logits[layer_idx]).to(dtype=xv.dtype)
            gates = gates.view(1, 1, n_heads, 1)
            xv = xv * gates
            x_out = xv.view(B, T, D)
            return (x_out,) + args[1:]
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def gates_01(self):
        return torch.sigmoid(self.G_logits).detach().cpu().numpy()


# ── Training ──────────────────────────────────────────────────────────────────

def encode_example(tokenizer, prompt, target, device):
    full = prompt + target
    enc_full = tokenizer(full, return_tensors="pt").to(device)
    enc_prompt = tokenizer(prompt, return_tensors="pt")
    prompt_len = enc_prompt["input_ids"].shape[1]
    return enc_full["input_ids"], prompt_len


def step(model, tokenizer, gate_bank, data, optimizer, lambd, device):
    """One epoch over the dataset."""
    losses = []
    for ex in data:
        input_ids, prompt_len = encode_example(tokenizer, ex["prompt"], ex["target"], device)

        outputs = model(input_ids=input_ids, return_dict=True)
        logits = outputs.logits[:, :-1].float()
        targets = input_ids[:, 1:]

        # Mask: only NLL on target tokens
        T = targets.shape[1]
        mask = torch.zeros(1, T, dtype=torch.bool, device=device)
        # target tokens are at positions [prompt_len-1 ... T-1] in the shifted target
        mask[:, prompt_len - 1:] = True

        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape)
        nll = (nll * mask.float()).sum() / mask.float().sum().clamp(min=1)

        # Regularization: -lambda * sum sigmoid^{-1}(G_l,h)
        # σ^{-1}(σ(x)) = x; we use clipped logits to mirror "clipped inverse-sigmoid"
        G_clipped = torch.clamp(gate_bank.G_logits, -10.0, 10.0)
        reg = -lambd * G_clipped.sum()

        loss = nll + reg
        optimizer.zero_grad()
        loss.backward()
        # Clip to handle any rare bf16 instability
        torch.nn.utils.clip_grad_norm_([gate_bank.G_logits], max_norm=10.0)
        optimizer.step()
        losses.append((nll.item(), reg.item(), loss.item()))

    return losses


def fit_phase(model, tokenizer, init_logits, lambd, n_steps, lr, label):
    cfg = model.config.text_config
    n_layers = cfg.num_hidden_layers
    n_heads  = cfg.num_attention_heads
    d_head   = cfg.head_dim if hasattr(cfg, "head_dim") else cfg.hidden_size // n_heads
    device = get_input_device(model)

    gate_bank = GateBank(model, n_layers, n_heads, d_head, init_logits=init_logits)
    optimizer = torch.optim.Adam([gate_bank.G_logits], lr=lr)

    print(f"\n[{label}] λ={lambd}, lr={lr}, n_steps={n_steps}", flush=True)
    t0 = time.time()
    try:
        for step_idx in range(n_steps):
            losses = step(model, tokenizer, gate_bank, DATA, optimizer, lambd, device)
            if step_idx % 5 == 0 or step_idx == n_steps - 1:
                avg_nll = sum(l[0] for l in losses) / len(losses)
                G_mean = torch.sigmoid(gate_bank.G_logits).mean().item()
                G_std  = torch.sigmoid(gate_bank.G_logits).std().item()
                elapsed = time.time() - t0
                print(f"  [{label}] step {step_idx:3d}/{n_steps}  "
                      f"avg_nll={avg_nll:.3f}  G_mean={G_mean:.3f}  "
                      f"G_std={G_std:.3f}  t={elapsed:.1f}s", flush=True)
        gates = gate_bank.gates_01()
        final_logits = gate_bank.G_logits.detach().clone()
    finally:
        gate_bank.remove()
        gc.collect()
        torch.cuda.empty_cache()

    return gates, final_logits


def main():
    torch.manual_seed(SEED)

    model, tokenizer, n_layers, n_heads, d_head = load_model()

    # Phase 0: warmup with lambda=0 to share init
    print("\n" + "=" * 60)
    print("Phase 0: lambda=0 warmup (shared init)")
    print("=" * 60, flush=True)
    _, warm_logits = fit_phase(model, tokenizer, None, lambd=0.0,
                                n_steps=N_STEPS_WARMUP, lr=LR, label="warm")

    # Phase 1: G+ (retention pressure, lambda > 0)
    print("\n" + "=" * 60)
    print("Phase 1: G+  (lambda > 0, retention pressure)")
    print("=" * 60, flush=True)
    G_plus, _ = fit_phase(model, tokenizer, warm_logits, lambd=LAMBDA_POS,
                           n_steps=N_STEPS_MAIN, lr=LR, label="G+")

    # Phase 2: G- (removal pressure, lambda < 0)
    print("\n" + "=" * 60)
    print("Phase 2: G-  (lambda < 0, removal pressure)")
    print("=" * 60, flush=True)
    G_minus, _ = fit_phase(model, tokenizer, warm_logits, lambd=LAMBDA_NEG,
                            n_steps=N_STEPS_MAIN, lr=LR, label="G-")

    # ── Save ──
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", RUN_NAME)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gates.json")
    with open(out_path, "w") as f:
        json.dump({
            "model_name":  MODEL_NAME,
            "n_layers":    n_layers,
            "n_heads":     n_heads,
            "d_head":      d_head,
            "n_data":      len(DATA),
            "n_steps_warmup": N_STEPS_WARMUP,
            "n_steps_main":   N_STEPS_MAIN,
            "lr":          LR,
            "lambda_pos":  LAMBDA_POS,
            "lambda_neg":  LAMBDA_NEG,
            "seed":        SEED,
            "G_plus":      G_plus.tolist(),
            "G_minus":     G_minus.tolist(),
            "facilitating_score":  G_minus.tolist(),  # high G^-
            "interfering_score":   (1.0 - G_plus).tolist(),
            "irrelevant_score":    (G_plus * (1.0 - G_minus)).tolist(),
        }, f, indent=2)

    # Quick analysis: top-K facilitating heads
    print("\n" + "=" * 60)
    print("Top-10 facilitating heads (high G-):")
    print("=" * 60)
    G_minus_flat = [(layer, head, G_minus[layer, head])
                    for layer in range(n_layers) for head in range(n_heads)]
    G_minus_flat.sort(key=lambda x: -x[2])
    for layer, head, score in G_minus_flat[:10]:
        print(f"  L{layer:2d} H{head:2d}  G-={score:.3f}  G+={G_plus[layer, head]:.3f}")

    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
