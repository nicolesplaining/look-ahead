"""
Component-level path patching and ablations for Gemma3-27B-IT.

Complements residual-stream patching by pinpointing specific attention heads
and MLPs in the critical window (layers 31-41) that mediate the newline-position
rhyme-planning handoff (peak effect at layer 33 in residual-stream experiments).

Metric: sampling-based rhyme rate (N=20 per combination), identical to the
        existing residual-stream experiments for direct comparability.

Four experiments
────────────────
A. Head path patching : replace head h's pre-o_proj slice at newline pos
                        with the corrupt run's value → measure corrupt_rhyme_rate.
                        → which heads are SUFFICIENT to steer toward corrupt rhyme?

B. MLP  path patching : replace layer L's MLP output at newline pos with corrupt
                        value → measure corrupt_rhyme_rate.
                        → which MLPs are SUFFICIENT?

C. Head ablation      : zero out head h's pre-o_proj contribution at newline pos
                        in the clean run → measure clean_rhyme_rate.
                        → which heads are NECESSARY for clean rhyme prediction?

D. MLP  ablation      : zero out layer L's MLP output at newline pos in clean run
                        → measure clean_rhyme_rate.
                        → which MLPs are NECESSARY?

Results: JSON + 4-panel figure (2 heatmaps, 2 bar charts).

Prompt pair: fright/fear  (same as GEMMA3_PER_LAYER residual-stream results,
                           enabling direct comparison)

Estimated runtime: ~6 hours on GH200 at N=20
"""

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pronouncing
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

# ── Config ──────────────────────────────────────────────────────────────────────

MODEL_NAME     = "google/gemma-3-27b-it"
CLEAN_PROMPT   = "A rhyming couplet:\nShe felt a sudden sense of fright,\nand hoped that"
CORRUPT_PROMPT = "A rhyming couplet:\nShe felt a sudden sense of fear,\nand hoped that"
CLEAN_RHYME_WORD   = "fright"
CORRUPT_RHYME_WORD = "fear"

SAMPLING_N    = 20
SAMPLING_TEMP = 0.8
MAX_NEW_TOKENS = 20

# Sweep the known critical window plus a few control layers on each side.
LAYER_RANGE = list(range(27, 46))   # layers 27–45 (critical window: 31–41)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results", "gemma3_27b_component_patching",
)

# ── Rhyme Checking ───────────────────────────────────────────────────────────────

def _rhyme_score(w1: str, w2: str):
    p1 = pronouncing.phones_for_word(w1.lower().strip())
    p2 = pronouncing.phones_for_word(w2.lower().strip())
    if not p1 or not p2:
        return None
    rp1 = pronouncing.rhyming_part(p1[0])
    rp2 = pronouncing.rhyming_part(p2[0])
    return (rp1 == rp2) if (rp1 and rp2) else None

def last_word(text: str) -> str:
    for w in reversed(text.split()):
        cleaned = w.strip(".,!?\"'—;: ")
        if cleaned.isalpha():
            return cleaned.lower()
    return ""

def word_before_nth_newline(text: str, n: int) -> str:
    if n <= 0:
        return ""
    newline_positions = [i for i, ch in enumerate(text) if ch == "\n"]
    if len(newline_positions) < n:
        return ""
    end   = newline_positions[n - 1]
    start = newline_positions[n - 2] + 1 if n >= 2 else 0
    return last_word(text[start:end])

def extract_rhyme_word(full_text: str, prompt: str) -> str:
    target_newline_index = prompt.count("\n") + 1
    rhyme_word = word_before_nth_newline(full_text, target_newline_index)
    if rhyme_word:
        return rhyme_word
    if full_text.startswith(prompt):
        return last_word(full_text[len(prompt):])
    return last_word(full_text)

def rhyme_rate(completions: list, prompt: str, rhyme_word: str) -> float:
    hits = sum(
        1 for c in completions
        if _rhyme_score(extract_rhyme_word(c, prompt), rhyme_word) is True
    )
    return hits / len(completions) if completions else 0.0

# ── Model Loading ────────────────────────────────────────────────────────────────

def load_model():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    tcfg = model.config.text_config
    print(
        f"Loaded. Layers: {tcfg.num_hidden_layers} | "
        f"Q-heads: {tcfg.num_attention_heads} | "
        f"KV-heads: {tcfg.num_key_value_heads} | "
        f"head_dim: {tcfg.head_dim}"
    )
    return model, tokenizer

def get_device(model):
    return model.model.language_model.embed_tokens.weight.device

# ── Generation ───────────────────────────────────────────────────────────────────

def generate_text(model, tokenizer, prompt: str) -> str:
    device = get_device(model)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=SAMPLING_TEMP,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def sample_completions(model, tokenizer, prompt: str, n: int, desc: str = "") -> list:
    return [generate_text(model, tokenizer, prompt) for _ in tqdm(range(n), desc=desc, leave=False)]

# ── Position Finding ─────────────────────────────────────────────────────────────

def find_newline_pos(tokenizer, prompt: str) -> int:
    """Token index of the last newline character in prompt."""
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    last_nl_char = max(i for i, ch in enumerate(prompt) if ch == "\n")
    pos = next(
        (i for i, (s, e) in enumerate(enc["offset_mapping"]) if s <= last_nl_char < e),
        None,
    )
    if pos is None:
        raise ValueError("No token covers the last newline.")
    return pos

# ── Activation Caching ───────────────────────────────────────────────────────────

def cache_pre_oproj(model, tokenizer, prompt: str, patch_pos: int, layers: list) -> dict:
    """Cache the input to each layer's self_attn.o_proj at patch_pos."""
    lm_layers = model.model.language_model.layers
    device = get_device(model)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    cached = {}
    handles = []

    def make_hook(idx):
        def hook(module, args):
            x = args[0]    # [batch, seq, num_q_heads * head_dim]
            if x.shape[1] > patch_pos:
                cached[idx] = x[:, patch_pos, :].detach().clone().cpu()
        return hook

    for idx in layers:
        handles.append(lm_layers[idx].self_attn.o_proj.register_forward_pre_hook(make_hook(idx)))
    try:
        with torch.no_grad():
            model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    missing = [idx for idx in layers if idx not in cached]
    if missing:
        raise RuntimeError(f"Pre-o_proj cache missing for layers: {missing}")
    return cached

def cache_mlp_outputs(model, tokenizer, prompt: str, patch_pos: int, layers: list) -> dict:
    """Cache each layer's MLP output at patch_pos."""
    lm_layers = model.model.language_model.layers
    device = get_device(model)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    cached = {}
    handles = []

    def make_hook(idx):
        def hook(module, inp, out):
            tensor = out[0] if isinstance(out, tuple) else out
            if tensor.shape[1] > patch_pos:
                cached[idx] = tensor[:, patch_pos, :].detach().clone().cpu()
        return hook

    for idx in layers:
        handles.append(lm_layers[idx].mlp.register_forward_hook(make_hook(idx)))
    try:
        with torch.no_grad():
            model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    missing = [idx for idx in layers if idx not in cached]
    if missing:
        raise RuntimeError(f"MLP output cache missing for layers: {missing}")
    return cached

# ── Patch / Ablate Context Managers ─────────────────────────────────────────────

@contextmanager
def patch_head_oproj(model, layer_idx, head_idx, patch_pos, corrupt_vec, head_dim):
    """Replace head h's slice of the pre-o_proj tensor at patch_pos with corrupt values."""
    s, e = head_idx * head_dim, (head_idx + 1) * head_dim
    op = model.model.language_model.layers[layer_idx].self_attn.o_proj

    def hook(module, args):
        x = args[0].clone()
        if x.shape[1] > patch_pos:
            x[:, patch_pos, s:e] = corrupt_vec[0, s:e].to(x.device, dtype=x.dtype)
        return (x,) + args[1:]

    handle = op.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()

@contextmanager
def patch_mlp_output(model, layer_idx, patch_pos, corrupt_vec):
    """Replace layer L's MLP output at patch_pos with corrupt values."""
    mlp = model.model.language_model.layers[layer_idx].mlp

    def hook(module, inp, out):
        if isinstance(out, tuple):
            out = list(out)
            if out[0].shape[1] > patch_pos:
                out[0] = out[0].clone()
                out[0][:, patch_pos, :] = corrupt_vec[0].to(out[0].device, dtype=out[0].dtype)
            return tuple(out)
        if out.shape[1] > patch_pos:
            out = out.clone()
            out[:, patch_pos, :] = corrupt_vec[0].to(out.device, dtype=out.dtype)
        return out

    handle = mlp.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()

@contextmanager
def ablate_head_oproj(model, layer_idx, head_idx, patch_pos, head_dim):
    """Zero out head h's contribution to the residual at patch_pos."""
    s, e = head_idx * head_dim, (head_idx + 1) * head_dim
    op = model.model.language_model.layers[layer_idx].self_attn.o_proj

    def hook(module, args):
        x = args[0].clone()
        if x.shape[1] > patch_pos:
            x[:, patch_pos, s:e] = 0.0
        return (x,) + args[1:]

    handle = op.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()

@contextmanager
def ablate_mlp_output(model, layer_idx, patch_pos):
    """Zero out layer L's MLP output at patch_pos."""
    mlp = model.model.language_model.layers[layer_idx].mlp

    def hook(module, inp, out):
        if isinstance(out, tuple):
            out = list(out)
            if out[0].shape[1] > patch_pos:
                out[0] = out[0].clone()
                out[0][:, patch_pos, :] = 0.0
            return tuple(out)
        if out.shape[1] > patch_pos:
            out = out.clone()
            out[:, patch_pos, :] = 0.0
        return out

    handle = mlp.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()

# ── Architecture Verification ────────────────────────────────────────────────────

def verify_architecture(model, tokenizer, patch_pos, num_heads, head_dim):
    """Sanity-check that the pre-o_proj tensor has the expected shape."""
    device = get_device(model)
    enc = tokenizer(CLEAN_PROMPT, return_tensors="pt").to(device)
    captured = {}

    def hook(module, args):
        captured["shape"] = tuple(args[0].shape)

    handle = model.model.language_model.layers[33].self_attn.o_proj.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model(**enc, use_cache=False)
    finally:
        handle.remove()

    seq_len = enc["input_ids"].shape[1]
    expected = (1, seq_len, num_heads * head_dim)
    assert captured["shape"] == expected, (
        f"Pre-o_proj shape mismatch: expected {expected}, got {captured['shape']}"
    )
    print(f"Architecture verified: pre-o_proj shape = {captured['shape']}")

# ── Main ─────────────────────────────────────────────────────────────────────────

def run():
    model, tokenizer = load_model()
    tcfg      = model.config.text_config
    num_heads = tcfg.num_attention_heads
    head_dim  = tcfg.head_dim

    patch_pos = find_newline_pos(tokenizer, CORRUPT_PROMPT)
    print(f"Newline patch position in corrupt prompt: {patch_pos}")

    verify_architecture(model, tokenizer, patch_pos, num_heads, head_dim)

    n_head_combos = len(LAYER_RANGE) * num_heads
    print(
        f"\nSweep: {len(LAYER_RANGE)} layers × {num_heads} heads = {n_head_combos} head "
        f"combos + {len(LAYER_RANGE)} MLP layers, N={SAMPLING_N} samples each\n"
    )

    # ── Baselines (unpatched) ─────────────────────────────────────────────────
    print("── Baselines ──")
    bl_completions = sample_completions(model, tokenizer, CLEAN_PROMPT, SAMPLING_N, "Baseline clean")
    bl_clean_rate   = rhyme_rate(bl_completions, CLEAN_PROMPT, CLEAN_RHYME_WORD)
    bl_corrupt_rate = rhyme_rate(bl_completions, CLEAN_PROMPT, CORRUPT_RHYME_WORD)
    print(f"  Clean prompt → clean_rate={bl_clean_rate:.2f}, corrupt_rate={bl_corrupt_rate:.2f}")

    # ── Cache corrupt activations ─────────────────────────────────────────────
    print(f"\nCaching corrupt activations at pos={patch_pos} "
          f"(layers {LAYER_RANGE[0]}–{LAYER_RANGE[-1]})...")
    corrupt_pre_oproj = cache_pre_oproj(
        model, tokenizer, CORRUPT_PROMPT, patch_pos, LAYER_RANGE
    )
    corrupt_mlp = cache_mlp_outputs(
        model, tokenizer, CORRUPT_PROMPT, patch_pos, LAYER_RANGE
    )

    # ── A: Head path patching ─────────────────────────────────────────────────
    print(f"\n── A: Head path patching ──")
    head_patch: dict = {}
    for layer_idx in tqdm(LAYER_RANGE, desc="  Layers (A)"):
        corrupt_vec = corrupt_pre_oproj[layer_idx]
        head_patch[layer_idx] = {}
        for head_idx in range(num_heads):
            with patch_head_oproj(model, layer_idx, head_idx, patch_pos, corrupt_vec, head_dim):
                completions = sample_completions(
                    model, tokenizer, CLEAN_PROMPT, SAMPLING_N,
                    desc=f"L{layer_idx}·H{head_idx}",
                )
            head_patch[layer_idx][head_idx] = {
                "corrupt_rhyme_rate": rhyme_rate(completions, CLEAN_PROMPT, CORRUPT_RHYME_WORD),
                "clean_rhyme_rate":   rhyme_rate(completions, CLEAN_PROMPT, CLEAN_RHYME_WORD),
            }

    # ── B: MLP path patching ──────────────────────────────────────────────────
    print(f"\n── B: MLP path patching ──")
    mlp_patch: dict = {}
    for layer_idx in tqdm(LAYER_RANGE, desc="  Layers (B)"):
        with patch_mlp_output(model, layer_idx, patch_pos, corrupt_mlp[layer_idx]):
            completions = sample_completions(
                model, tokenizer, CLEAN_PROMPT, SAMPLING_N, desc=f"L{layer_idx}",
            )
        mlp_patch[layer_idx] = {
            "corrupt_rhyme_rate": rhyme_rate(completions, CLEAN_PROMPT, CORRUPT_RHYME_WORD),
            "clean_rhyme_rate":   rhyme_rate(completions, CLEAN_PROMPT, CLEAN_RHYME_WORD),
        }

    # ── C: Head ablation (clean run) ──────────────────────────────────────────
    print(f"\n── C: Head ablation (clean run) ──")
    head_ablate: dict = {}
    for layer_idx in tqdm(LAYER_RANGE, desc="  Layers (C)"):
        head_ablate[layer_idx] = {}
        for head_idx in range(num_heads):
            with ablate_head_oproj(model, layer_idx, head_idx, patch_pos, head_dim):
                completions = sample_completions(
                    model, tokenizer, CLEAN_PROMPT, SAMPLING_N,
                    desc=f"L{layer_idx}·H{head_idx}",
                )
            head_ablate[layer_idx][head_idx] = {
                "corrupt_rhyme_rate": rhyme_rate(completions, CLEAN_PROMPT, CORRUPT_RHYME_WORD),
                "clean_rhyme_rate":   rhyme_rate(completions, CLEAN_PROMPT, CLEAN_RHYME_WORD),
            }

    # ── D: MLP ablation (clean run) ───────────────────────────────────────────
    print(f"\n── D: MLP ablation (clean run) ──")
    mlp_ablate: dict = {}
    for layer_idx in tqdm(LAYER_RANGE, desc="  Layers (D)"):
        with ablate_mlp_output(model, layer_idx, patch_pos):
            completions = sample_completions(
                model, tokenizer, CLEAN_PROMPT, SAMPLING_N, desc=f"L{layer_idx}",
            )
        mlp_ablate[layer_idx] = {
            "corrupt_rhyme_rate": rhyme_rate(completions, CLEAN_PROMPT, CORRUPT_RHYME_WORD),
            "clean_rhyme_rate":   rhyme_rate(completions, CLEAN_PROMPT, CLEAN_RHYME_WORD),
        }

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)

    export = {
        "timestamp_utc":      datetime.now(timezone.utc).isoformat(),
        "model_name":         MODEL_NAME,
        "clean_prompt":       CLEAN_PROMPT,
        "corrupt_prompt":     CORRUPT_PROMPT,
        "clean_rhyme_word":   CLEAN_RHYME_WORD,
        "corrupt_rhyme_word": CORRUPT_RHYME_WORD,
        "newline_patch_pos":  patch_pos,
        "layer_range":        LAYER_RANGE,
        "num_q_heads":        num_heads,
        "head_dim":           head_dim,
        "sampling_n":         SAMPLING_N,
        "sampling_temp":      SAMPLING_TEMP,
        "baselines": {
            "clean_rhyme_rate":   bl_clean_rate,
            "corrupt_rhyme_rate": bl_corrupt_rate,
        },
        "head_path_patching": {
            str(L): {str(h): v for h, v in hd.items()}
            for L, hd in head_patch.items()
        },
        "mlp_path_patching":  {str(L): v for L, v in mlp_patch.items()},
        "head_ablation":      {
            str(L): {str(h): v for h, v in hd.items()}
            for L, hd in head_ablate.items()
        },
        "mlp_ablation":       {str(L): v for L, v in mlp_ablate.items()},
    }

    json_path = os.path.join(RESULTS_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"\nSaved to {json_path}")

    plot_results(export)
    print_top_candidates(export)
    return export

# ── Visualisation ────────────────────────────────────────────────────────────────

def plot_results(data: dict):
    import numpy as np

    layer_range = data["layer_range"]
    num_heads   = data["num_q_heads"]
    bl_corrupt  = data["baselines"]["corrupt_rhyme_rate"]
    bl_clean    = data["baselines"]["clean_rhyme_rate"]

    def head_matrix(key: str, rate_key: str) -> np.ndarray:
        m = np.zeros((len(layer_range), num_heads))
        for i, L in enumerate(layer_range):
            for h in range(num_heads):
                m[i, h] = data[key][str(L)][str(h)][rate_key]
        return m

    crit_idx_start = layer_range.index(31) if 31 in layer_range else None
    crit_idx_end   = layer_range.index(41) if 41 in layer_range else None

    def draw_window(ax):
        if crit_idx_start is None or crit_idx_end is None:
            return
        kw = dict(color="gold", lw=1.5, ls="--", alpha=0.8)
        ax.axhline(crit_idx_start - 0.5, **kw)
        ax.axhline(crit_idx_end   + 0.5, **kw)

    def draw_window_bar(ax):
        if crit_idx_start is None or crit_idx_end is None:
            return
        kw = dict(color="gold", lw=1.5, ls="--", alpha=0.8)
        ax.axhline(crit_idx_start - 0.5, **kw)
        ax.axhline(crit_idx_end   + 0.5, **kw)

    fig, axes = plt.subplots(2, 2, figsize=(24, 14))

    # ── A: head path patching ──
    matrix_a = head_matrix("head_path_patching", "corrupt_rhyme_rate") - bl_corrupt
    ax = axes[0, 0]
    vmax_a = max(matrix_a.max(), 0.05)
    im_a = ax.imshow(matrix_a, cmap="Reds", aspect="auto", vmin=0, vmax=vmax_a)
    ax.set_yticks(range(len(layer_range)))
    ax.set_yticklabels(layer_range, fontsize=8)
    ax.set_xticks(range(0, num_heads, 2))
    ax.set_xticklabels(range(0, num_heads, 2), fontsize=7)
    ax.set_xlabel("Head index")
    ax.set_ylabel("Layer")
    ax.set_title(
        f"A. Head path patching\n"
        f"Δ corrupt rhyme rate  (baseline = {bl_corrupt:.2f})",
        fontsize=10,
    )
    draw_window(ax)
    plt.colorbar(im_a, ax=ax, fraction=0.03)

    # ── B: MLP path patching ──
    ax = axes[0, 1]
    mlp_p_vals = [data["mlp_path_patching"][str(L)]["corrupt_rhyme_rate"] - bl_corrupt
                  for L in layer_range]
    colors_b = ["tomato" if 31 <= L <= 41 else "lightgray" for L in layer_range]
    ax.barh(range(len(layer_range)), mlp_p_vals, color=colors_b)
    ax.set_yticks(range(len(layer_range)))
    ax.set_yticklabels(layer_range, fontsize=8)
    ax.set_xlabel("Δ corrupt rhyme rate")
    ax.set_title(
        f"B. MLP path patching\n"
        f"Δ corrupt rhyme rate  (baseline = {bl_corrupt:.2f})",
        fontsize=10,
    )
    ax.axvline(0, color="black", lw=0.5)
    draw_window_bar(ax)
    for i, v in enumerate(mlp_p_vals):
        if abs(v) > 0.02:
            ax.text(v + (0.005 if v >= 0 else -0.005), i,
                    f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=7)

    # ── C: head ablation ──
    matrix_c = head_matrix("head_ablation", "clean_rhyme_rate") - bl_clean
    ax = axes[1, 0]
    vmin_c = min(matrix_c.min(), -0.05)
    im_c = ax.imshow(matrix_c, cmap="Blues_r", aspect="auto", vmin=vmin_c, vmax=0)
    ax.set_yticks(range(len(layer_range)))
    ax.set_yticklabels(layer_range, fontsize=8)
    ax.set_xticks(range(0, num_heads, 2))
    ax.set_xticklabels(range(0, num_heads, 2), fontsize=7)
    ax.set_xlabel("Head index")
    ax.set_ylabel("Layer")
    ax.set_title(
        f"C. Head ablation (clean run)\n"
        f"Δ clean rhyme rate  (baseline = {bl_clean:.2f})",
        fontsize=10,
    )
    draw_window(ax)
    plt.colorbar(im_c, ax=ax, fraction=0.03)

    # ── D: MLP ablation ──
    ax = axes[1, 1]
    mlp_a_vals = [data["mlp_ablation"][str(L)]["clean_rhyme_rate"] - bl_clean
                  for L in layer_range]
    colors_d = ["steelblue" if 31 <= L <= 41 else "lightgray" for L in layer_range]
    ax.barh(range(len(layer_range)), mlp_a_vals, color=colors_d)
    ax.set_yticks(range(len(layer_range)))
    ax.set_yticklabels(layer_range, fontsize=8)
    ax.set_xlabel("Δ clean rhyme rate")
    ax.set_title(
        f"D. MLP ablation (clean run)\n"
        f"Δ clean rhyme rate  (baseline = {bl_clean:.2f})",
        fontsize=10,
    )
    ax.axvline(0, color="black", lw=0.5)
    draw_window_bar(ax)
    for i, v in enumerate(mlp_a_vals):
        if abs(v) > 0.02:
            ax.text(v + (0.005 if v >= 0 else -0.005), i,
                    f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=7)

    fig.suptitle(
        "Gemma3-27B — Component-level analysis at newline position\n"
        f"Prompt pair: {CORRUPT_RHYME_WORD} → {CLEAN_RHYME_WORD}   "
        f"N={data['sampling_n']}   "
        f"(dashed lines = known critical window layers 31–41)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "component_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out}")

# ── Console Summary ───────────────────────────────────────────────────────────────

def print_top_candidates(data: dict, top_n: int = 10):
    bl_corrupt  = data["baselines"]["corrupt_rhyme_rate"]
    bl_clean    = data["baselines"]["clean_rhyme_rate"]
    num_heads   = data["num_q_heads"]
    layer_range = data["layer_range"]

    print(f"\n{'─'*60}")
    print(f"Top {top_n} heads by path-patching  (Δ corrupt rhyme rate, base={bl_corrupt:.2f}):")
    deltas = []
    for L in layer_range:
        for h in range(num_heads):
            d = data["head_path_patching"][str(L)][str(h)]["corrupt_rhyme_rate"] - bl_corrupt
            deltas.append((d, L, h))
    for d, L, h in sorted(deltas, reverse=True)[:top_n]:
        print(f"  L{L:2d}·H{h:2d}  Δ = {d:+.2f}")

    print(f"\nTop {top_n} heads by ablation  (Δ clean rhyme rate, base={bl_clean:.2f}):")
    deltas = []
    for L in layer_range:
        for h in range(num_heads):
            d = data["head_ablation"][str(L)][str(h)]["clean_rhyme_rate"] - bl_clean
            deltas.append((d, L, h))
    for d, L, h in sorted(deltas)[:top_n]:
        print(f"  L{L:2d}·H{h:2d}  Δ = {d:+.2f}")

    print(f"\nMLP path patching (top 5 by Δ corrupt rhyme rate):")
    mlp_d = [(data["mlp_path_patching"][str(L)]["corrupt_rhyme_rate"] - bl_corrupt, L)
             for L in layer_range]
    for d, L in sorted(mlp_d, reverse=True)[:5]:
        print(f"  L{L:2d}  Δ = {d:+.2f}")

    print(f"\nMLP ablation (top 5 by Δ clean rhyme rate):")
    mlp_d = [(data["mlp_ablation"][str(L)]["clean_rhyme_rate"] - bl_clean, L)
             for L in layer_range]
    for d, L in sorted(mlp_d)[:5]:
        print(f"  L{L:2d}  Δ = {d:+.2f}")

    print(f"{'─'*60}\n")


if __name__ == "__main__":
    run()
