"""Minimal CHG diagnostic. Test forward with and without hook to isolate NaN."""

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForConditionalGeneration

MODEL_NAME = "google/gemma-3-27b-it"
PROMPT = "A rhyming couplet:\nThe empty house was filled with silent doom,\nwhen suddenly they"

print("Loading...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_NAME, quantization_config=bnb, device_map="auto", trust_remote_code=True)
for p in model.parameters():
    p.requires_grad_(False)
model.eval()

cfg = model.config.text_config
print(f"Layers={cfg.num_hidden_layers}  Heads={cfg.num_attention_heads}  d_head={cfg.head_dim}")

# Test 1: clean forward
device = model.model.language_model.embed_tokens.weight.device
ids = tok(PROMPT, return_tensors="pt").to(device)["input_ids"]
with torch.no_grad():
    out = model(input_ids=ids).logits
print(f"Test 1 (no hook): logits shape={out.shape}  finite_pct={torch.isfinite(out).float().mean().item()*100:.1f}%")
print(f"  logits[0,-1,:5]={out[0,-1,:5].tolist()}")

# Test 2: install identity hook
n_heads = cfg.num_attention_heads
d_head  = cfg.head_dim
G = torch.full((cfg.num_hidden_layers, n_heads), 4.0, device=device, requires_grad=True)

def make_hook(idx):
    def hook(mod, args):
        x = args[0]
        print(f"  hook L{idx}: input shape={x.shape}  finite={torch.isfinite(x).all().item()}")
        B, T, D = x.shape
        xv = x.view(B, T, n_heads, d_head)
        gates = torch.sigmoid(G[idx]).to(dtype=xv.dtype).view(1,1,n_heads,1)
        xv = xv * gates
        return (xv.view(B, T, D),) + args[1:]
    return hook

# install on first 3 layers only, to limit prints
handles = []
for i, layer in enumerate(model.model.language_model.layers[:3]):
    handles.append(layer.self_attn.o_proj.register_forward_pre_hook(make_hook(i)))

print("\nTest 2 (gates=sigmoid(4)~0.98 on first 3 layers):")
with torch.no_grad():
    out = model(input_ids=ids).logits
print(f"  logits finite_pct={torch.isfinite(out).float().mean().item()*100:.1f}%")
print(f"  logits[0,-1,:5]={out[0,-1,:5].tolist()}")

for h in handles:
    h.remove()
