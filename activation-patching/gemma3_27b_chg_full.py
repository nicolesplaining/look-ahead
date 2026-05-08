"""
Full CHG run for Gemma-3-27B-IT on rhyming-couplet completion.

Differences from smoke test:
  - 100 couplets (sampled from poems-original.jsonl) instead of 5
  - 5 epochs over full dataset per phase
  - Weaker reg (lambda=±0.005) — gentle enough that NLL can drive head selection
  - 3 seeds (0, 1, 2) for ranking stability
  - NLL masked to second-line tokens only (prompt tokens excluded from loss)

Outputs to results/gemma3_27b_chg_full/:
  - gates_seed{0,1,2}.json    per-seed G+, G- matrices
  - aggregate.json             mean across seeds + facilitating/interfering/irrelevant scores

Then run gemma3_27b_chg_validate.py (separate script) to take top-k
CHG-facilitating heads and run simultaneous-patching, comparing to the
existing top-5 attention-weight heads.
"""

import gc
import json
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "google/gemma-3-27b-it"
RUN_NAME   = "gemma3_27b_chg_full"

DATA_PATH = os.path.expanduser("~/look-ahead-3/poem/data/poems-original.jsonl")
N_DATA    = 100

N_STEPS_WARMUP = 30
N_EPOCHS_MAIN  = 5
LR             = 0.05
LAMBDA_POS     = 0.005
LAMBDA_NEG     = -0.005

SEEDS = [0, 1, 2]


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data(seed):
    """Return list of (prompt, target_continuation) tuples. Couplets only."""
    items = []
    with open(DATA_PATH) as f:
        for line in f:
            d = json.loads(line)
            text = d["text"]
            # Format: "A rhyming couplet:\nFIRST_LINE,\nSECOND_LINE\n"
            lines = text.split("\n")
            if len(lines) < 3 or not lines[0].startswith("A rhyming couplet"):
                continue
            prompt = lines[0] + "\n" + lines[1] + "\n"  # ends after second newline
            target = lines[2]
            if not target.strip():
                continue
            items.append({"prompt": prompt, "target": target})
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:N_DATA]


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model():
    print(f"Loading {MODEL_NAME} (bf16)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    for p in model.parameters():
        p.requires_grad_(False)

    cfg = model.config.text_config
    n_layers = cfg.num_hidden_layers
    n_heads  = cfg.num_attention_heads
    d_head   = cfg.head_dim if hasattr(cfg, "head_dim") else cfg.hidden_size // n_heads
    print(f"Loaded. Layers={n_layers}  Heads/layer={n_heads}  d_head={d_head}", flush=True)
    return model, tokenizer, n_layers, n_heads, d_head


def get_layers(model):
    return model.model.language_model.layers


def get_input_device(model):
    return model.model.language_model.embed_tokens.weight.device


# ── Gates ──────────────────────────────────────────────────────────────────────

class GateBank:
    def __init__(self, model, n_layers, n_heads, d_head, init_logits=None):
        self.n_layers = n_layers
        self.n_heads  = n_heads
        self.d_head   = d_head
        device = get_input_device(model)
        # Init near-identity (sigmoid(4)~=0.98)
        if init_logits is None:
            init = 4.0 * torch.ones(n_layers, n_heads, device=device, dtype=torch.float32)
        else:
            init = init_logits.clone().to(device)
        self.G_logits = nn.Parameter(init)
        self.handles = []
        self._install(model)

    def _install(self, model):
        for i, layer in enumerate(get_layers(model)):
            o_proj = layer.self_attn.o_proj
            self.handles.append(o_proj.register_forward_pre_hook(self._make_hook(i)))

    def _make_hook(self, layer_idx):
        n_heads = self.n_heads
        d_head  = self.d_head
        def hook(module, args):
            x = args[0]
            B, T, D = x.shape
            xv = x.view(B, T, n_heads, d_head)
            gates = torch.sigmoid(self.G_logits[layer_idx]).to(dtype=xv.dtype)
            gates = gates.view(1, 1, n_heads, 1)
            xv = xv * gates
            return (xv.view(B, T, D),) + args[1:]
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


def step_one(model, tokenizer, gate_bank, ex, optimizer, lambd, device):
    input_ids, prompt_len = encode_example(tokenizer, ex["prompt"], ex["target"], device)
    out = model(input_ids=input_ids, return_dict=True)
    logits  = out.logits[:, :-1].float()
    targets = input_ids[:, 1:]

    T = targets.shape[1]
    mask = torch.zeros(1, T, dtype=torch.bool, device=device)
    mask[:, prompt_len - 1:] = True

    nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).reshape(targets.shape)
    nll = (nll * mask.float()).sum() / mask.float().sum().clamp(min=1)

    G_clipped = torch.clamp(gate_bank.G_logits, -10.0, 10.0)
    reg = -lambd * G_clipped.sum()

    loss = nll + reg
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([gate_bank.G_logits], max_norm=10.0)
    optimizer.step()
    return nll.item()


def fit_phase(model, tokenizer, data, init_logits, lambd, n_epochs, lr, label, n_warmup_steps=None):
    cfg = model.config.text_config
    n_layers = cfg.num_hidden_layers
    n_heads  = cfg.num_attention_heads
    d_head   = cfg.head_dim if hasattr(cfg, "head_dim") else cfg.hidden_size // n_heads
    device = get_input_device(model)

    bank = GateBank(model, n_layers, n_heads, d_head, init_logits=init_logits)
    opt  = torch.optim.Adam([bank.G_logits], lr=lr)

    print(f"\n[{label}] λ={lambd}  lr={lr}  data={len(data)}  "
          f"{'warmup_steps='+str(n_warmup_steps) if n_warmup_steps else 'epochs='+str(n_epochs)}", flush=True)
    t0 = time.time()
    try:
        if n_warmup_steps is not None:
            # Warmup: cycle through data for fixed step count
            losses = []
            for s in range(n_warmup_steps):
                ex = data[s % len(data)]
                losses.append(step_one(model, tokenizer, bank, ex, opt, lambd, device))
                if s % 10 == 0 or s == n_warmup_steps - 1:
                    G_mean = torch.sigmoid(bank.G_logits).mean().item()
                    G_std  = torch.sigmoid(bank.G_logits).std().item()
                    print(f"  [{label}] step {s:4d}/{n_warmup_steps}  "
                          f"avg_nll={sum(losses[-10:])/min(10,len(losses)):.3f}  "
                          f"G_mean={G_mean:.3f}  G_std={G_std:.3f}  "
                          f"t={time.time()-t0:.1f}s", flush=True)
        else:
            losses = []
            n_examples = len(data)
            for epoch in range(n_epochs):
                ep_losses = []
                # shuffle each epoch
                rng = random.Random(epoch * 7919)
                order = list(range(n_examples))
                rng.shuffle(order)
                for idx, j in enumerate(order):
                    ep_losses.append(step_one(model, tokenizer, bank, data[j], opt, lambd, device))
                    if idx % 25 == 0:
                        G_mean = torch.sigmoid(bank.G_logits).mean().item()
                        avg_nll = sum(ep_losses[-25:]) / min(25, len(ep_losses))
                        print(f"  [{label}] epoch {epoch} step {idx:3d}/{n_examples}  "
                              f"avg_nll={avg_nll:.3f}  G_mean={G_mean:.3f}  "
                              f"t={time.time()-t0:.1f}s", flush=True)
                losses.extend(ep_losses)

        gates = bank.gates_01()
        final_logits = bank.G_logits.detach().clone()
    finally:
        bank.remove()
        gc.collect()
        torch.cuda.empty_cache()

    return gates, final_logits


def run_seed(model, tokenizer, seed, out_dir):
    """Run all 3 phases for one seed; save per-seed gates."""
    seed_path = os.path.join(out_dir, f"gates_seed{seed}.json")
    if os.path.exists(seed_path):
        print(f"\nSeed {seed} already done, loading.")
        with open(seed_path) as f:
            return json.load(f)

    print(f"\n{'#'*60}\n# SEED {seed}\n{'#'*60}", flush=True)
    torch.manual_seed(seed)
    random.seed(seed)

    data = load_data(seed)
    print(f"Loaded {len(data)} couplets", flush=True)

    print(f"\n=== Phase 0: warmup (λ=0) ===", flush=True)
    _, warm_logits = fit_phase(model, tokenizer, data, None, lambd=0.0,
                                n_epochs=1, lr=LR, label=f"warm_s{seed}",
                                n_warmup_steps=N_STEPS_WARMUP)

    print(f"\n=== Phase 1: G+ (λ={LAMBDA_POS}) ===", flush=True)
    G_plus, _ = fit_phase(model, tokenizer, data, warm_logits, lambd=LAMBDA_POS,
                           n_epochs=N_EPOCHS_MAIN, lr=LR, label=f"G+_s{seed}")

    print(f"\n=== Phase 2: G- (λ={LAMBDA_NEG}) ===", flush=True)
    G_minus, _ = fit_phase(model, tokenizer, data, warm_logits, lambd=LAMBDA_NEG,
                            n_epochs=N_EPOCHS_MAIN, lr=LR, label=f"G-_s{seed}")

    out = {
        "seed":       seed,
        "model_name": MODEL_NAME,
        "n_data":     len(data),
        "n_epochs":   N_EPOCHS_MAIN,
        "lr":         LR,
        "lambda_pos": LAMBDA_POS,
        "lambda_neg": LAMBDA_NEG,
        "G_plus":     G_plus.tolist(),
        "G_minus":    G_minus.tolist(),
    }
    with open(seed_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {seed_path}", flush=True)
    return out


def aggregate_seeds(out_dir, seeds):
    import numpy as np
    G_plus_list  = []
    G_minus_list = []
    for s in seeds:
        with open(os.path.join(out_dir, f"gates_seed{s}.json")) as f:
            d = json.load(f)
        G_plus_list.append(np.array(d["G_plus"]))
        G_minus_list.append(np.array(d["G_minus"]))
    G_plus  = np.stack(G_plus_list).mean(axis=0)   # [L, H]
    G_minus = np.stack(G_minus_list).mean(axis=0)
    facilitating = G_minus
    interfering  = 1.0 - G_plus
    irrelevant   = G_plus * (1.0 - G_minus)

    L, H = G_plus.shape
    # Top-K facilitating
    flat = [(layer, head, facilitating[layer, head]) for layer in range(L) for head in range(H)]
    flat.sort(key=lambda x: -x[2])

    out = {
        "model_name":        MODEL_NAME,
        "seeds":             seeds,
        "n_layers":          int(L),
        "n_heads":           int(H),
        "G_plus_mean":       G_plus.tolist(),
        "G_minus_mean":      G_minus.tolist(),
        "facilitating":      facilitating.tolist(),
        "interfering":       interfering.tolist(),
        "irrelevant":        irrelevant.tolist(),
        "top_facilitating":  [{"layer": l, "head": h, "score": float(s)} for l, h, s in flat[:20]],
    }
    agg_path = os.path.join(out_dir, "aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nAggregate saved to {agg_path}")

    print("\nTop-20 facilitating heads (averaged over seeds):")
    for l, h, s in flat[:20]:
        print(f"  L{l:2d} H{h:2d}  fac={s:.3f}  G+={G_plus[l,h]:.3f}  G-={G_minus[l,h]:.3f}")


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", RUN_NAME)
    os.makedirs(out_dir, exist_ok=True)

    model, tokenizer, _, _, _ = load_model()
    for seed in SEEDS:
        run_seed(model, tokenizer, seed, out_dir)
    aggregate_seeds(out_dir, SEEDS)


if __name__ == "__main__":
    main()
