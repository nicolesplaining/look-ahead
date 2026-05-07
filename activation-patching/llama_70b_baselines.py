"""
Activation-patching baseline experiments for Llama-3.1-70B-Instruct.

Two baseline patch modes (neither uses corrupt activations):
  1. zero_vector  — replace hidden state at patch position with zeros
  2. donor_prompt — replace with hidden state from an unrelated prompt at the
                    same absolute token position

Patches at i=-1 (last word token for Llama-3 due to its `,\\n` single-token
tokenization), all 80 layers, 5 prompt pairs, N=20 samples.

Mirrors baseline_experiment_qwen.py / baseline_experiment_gemma.py with
70B-specific 4-bit loading and batched sampling.

Results saved to results/LLAMA70B_BASELINE/<pair_id>/<patch_mode>/generations.json
"""

import gc
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone

import pronouncing
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME     = "meta-llama/Llama-3.1-70B-Instruct"
SAMPLING_N     = 20
SAMPLING_TEMP  = 0.7
MAX_NEW_TOKENS = 20
BATCH_SIZE     = 20

PATCH_OFFSET = -1  # Llama tokenizes ",\n" as one token; last word is at i=-1

DONOR_PROMPT = (
    "The weather outside is warm and sunny today, and the birds are singing.\n"
)

PROMPT_PAIRS = [
    {
        "pair_id":            "doom_dread",
        "clean_prompt":       "A rhyming couplet:\nThe empty house was filled with silent doom,\nwhen suddenly they",
        "corrupt_prompt":     "A rhyming couplet:\nThe empty house was filled with silent dread,\nwhen suddenly they",
        "clean_rhyme_word":   "doom",
        "corrupt_rhyme_word": "dread",
    },
    {
        "pair_id":            "bliss_joy",
        "clean_prompt":       "A rhyming couplet:\nThe children laughed in bliss,\nuntil they all",
        "corrupt_prompt":     "A rhyming couplet:\nThe children laughed in joy,\nuntil they all",
        "clean_rhyme_word":   "bliss",
        "corrupt_rhyme_word": "joy",
    },
    {
        "pair_id":            "dark_night",
        "clean_prompt":       "A rhyming couplet:\nShe wandered home alone into the dark,\nand then she",
        "corrupt_prompt":     "A rhyming couplet:\nShe wandered home alone into the night,\nand then she",
        "clean_rhyme_word":   "dark",
        "corrupt_rhyme_word": "night",
    },
    {
        "pair_id":            "grief_pain",
        "clean_prompt":       "A rhyming couplet:\nI never knew the depth of such grief,\nas though the",
        "corrupt_prompt":     "A rhyming couplet:\nI never knew the depth of such pain,\nas though the",
        "clean_rhyme_word":   "grief",
        "corrupt_rhyme_word": "pain",
    },
    {
        "pair_id":            "fright_fear",
        "clean_prompt":       "A rhyming couplet:\nShe felt a sudden sense of fright,\nand hoped that",
        "corrupt_prompt":     "A rhyming couplet:\nShe felt a sudden sense of fear,\nand hoped that",
        "clean_rhyme_word":   "fright",
        "corrupt_rhyme_word": "fear",
    },
]

# ── Rhyme helpers ──────────────────────────────────────────────────────────────

def _rhyme_score(w1, w2):
    p1 = pronouncing.phones_for_word(w1.lower().strip())
    p2 = pronouncing.phones_for_word(w2.lower().strip())
    if not p1 or not p2:
        return None
    rp1 = pronouncing.rhyming_part(p1[0])
    rp2 = pronouncing.rhyming_part(p2[0])
    return (rp1 == rp2) if (rp1 and rp2) else None

def last_word(text):
    for w in reversed(text.split()):
        cleaned = w.strip(".,!?\"'—;: ")
        if cleaned.isalpha():
            return cleaned.lower()
    return ""

def word_before_nth_newline(text, n):
    if n <= 0:
        return ""
    nps = [i for i, ch in enumerate(text) if ch == "\n"]
    if len(nps) < n:
        return ""
    end   = nps[n - 1]
    start = nps[n - 2] + 1 if n >= 2 else 0
    return last_word(text[start:end])

def extract_rhyme_word(full_text, prompt):
    target = prompt.count("\n") + 1
    w = word_before_nth_newline(full_text, target)
    if w:
        return w
    if full_text.startswith(prompt):
        return last_word(full_text[len(prompt):])
    return last_word(full_text)

def rhyme_rate(completions, prompt, rhyme_word):
    hits = sum(1 for c in completions
               if _rhyme_score(extract_rhyme_word(c, prompt), rhyme_word) is True)
    return hits / len(completions) if completions else 0.0

# ── Model ──────────────────────────────────────────────────────────────────────

def load_model():
    print(f"Loading {MODEL_NAME} (4-bit)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto",
    )
    model.eval()
    gc.collect(); torch.cuda.empty_cache()
    print(f"Loaded. Layers: {model.config.num_hidden_layers}", flush=True)
    return model, tokenizer

def get_input_device(model): return model.model.embed_tokens.weight.device
def get_layers(model):       return model.model.layers
def get_n_layers(model):     return model.config.num_hidden_layers

# ── Generation ─────────────────────────────────────────────────────────────────

def generate_text(model, tokenizer, prompt, temperature):
    device = get_input_device(model)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    del out, enc; gc.collect(); torch.cuda.empty_cache()
    return text

def sample_completions(model, tokenizer, prompt, n, temperature, batch_size=BATCH_SIZE):
    device = get_input_device(model)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    completions = []
    remaining = n
    while remaining > 0:
        b = min(remaining, batch_size)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True, temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=b,
            )
        for row in out:
            completions.append(tokenizer.decode(row, skip_special_tokens=True))
        del out
        remaining -= b
    del enc; gc.collect(); torch.cuda.empty_cache()
    return completions

# ── Position finding ───────────────────────────────────────────────────────────

def find_patch_pos(tokenizer, prompt, offset):
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    token_ids = enc["input_ids"]
    om = enc["offset_mapping"]
    nls = [i for i, ch in enumerate(prompt) if ch == "\n"]
    if len(nls) < 2:
        raise ValueError("Need 2+ newlines")
    second_nl_char = nls[1]
    second_nl_tok = next(
        (i for i, (s, e) in enumerate(om) if s <= second_nl_char < e),
        None,
    )
    if second_nl_tok is None:
        raise ValueError("Could not find newline token")
    patch_pos = second_nl_tok + offset
    tok_str = tokenizer.decode([token_ids[patch_pos]])
    return patch_pos, tok_str

# ── Activation caching ─────────────────────────────────────────────────────────

def cache_hidden_states_at_pos(model, tokenizer, prompt, patch_pos):
    layers = get_layers(model)
    device = get_input_device(model)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    cached = [None] * len(layers)
    handles = []
    def make(idx):
        def hook(mod, args):
            h = args[0]
            if h.shape[1] > patch_pos:
                cached[idx] = h[:, patch_pos, :].detach().clone()
        return hook
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(make(i)))
    try:
        with torch.no_grad():
            model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    if any(v is None for v in cached):
        raise RuntimeError("Some layers failed to cache")
    return cached

# ── Patching ───────────────────────────────────────────────────────────────────

@contextmanager
def patch_layer_at_pos(model, layer_idx, patch_pos, patch_vec):
    layer = get_layers(model)[layer_idx]
    def hook(mod, args):
        h = args[0]
        if h.shape[1] <= 1 or h.shape[1] <= patch_pos:
            return args
        out = h.clone()
        out[:, patch_pos, :] = patch_vec.to(out.device, dtype=out.dtype)
        return (out,) + args[1:]
    handle = layer.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()

# ── Single baseline run ────────────────────────────────────────────────────────

def run_baseline(model, tokenizer, pair, patch_mode, patch_vecs,
                 patch_pos, baseline_completions, baseline_clean_rate,
                 baseline_corrupt_rate, out_dir):
    json_path = os.path.join(out_dir, "generations.json")
    if os.path.exists(json_path):
        print(f"    [{patch_mode}] already done", flush=True)
        with open(json_path) as f:
            return json.load(f)

    clean_prompt       = pair["clean_prompt"]
    clean_rhyme_word   = pair["clean_rhyme_word"]
    corrupt_rhyme_word = pair["corrupt_rhyme_word"]
    n_layers           = get_n_layers(model)

    layer_results = []
    for layer_idx in tqdm(range(n_layers), desc=f"    layers [{patch_mode}]", leave=False):
        patch_vec = patch_vecs[layer_idx]
        with patch_layer_at_pos(model, layer_idx, patch_pos, patch_vec):
            completions = sample_completions(model, tokenizer, clean_prompt, SAMPLING_N, SAMPLING_TEMP)
        clean_rate   = rhyme_rate(completions, clean_prompt, clean_rhyme_word)
        corrupt_rate = rhyme_rate(completions, clean_prompt, corrupt_rhyme_word)
        layer_results.append({
            "layer":                 layer_idx,
            "completions":           completions,
            "clean_rhyme_rate":      clean_rate,
            "corrupt_rhyme_rate":    corrupt_rate,
            "baseline_clean_rate":   baseline_clean_rate,
            "baseline_corrupt_rate": baseline_corrupt_rate,
            "delta_corrupt_rate":    corrupt_rate - baseline_corrupt_rate,
        })
        gc.collect(); torch.cuda.empty_cache()

    best = max(layer_results, key=lambda r: r["corrupt_rhyme_rate"])
    print(f"    [{patch_mode}] best layer {best['layer']}: corrupt_rate={best['corrupt_rhyme_rate']:.3f} "
          f"(baseline={baseline_corrupt_rate:.3f})", flush=True)

    export = {
        "timestamp_utc":      datetime.now(timezone.utc).isoformat(),
        "model_name":         MODEL_NAME,
        "pair_id":            pair["pair_id"],
        "patch_mode":         patch_mode,
        "patch_offset":       PATCH_OFFSET,
        "patch_pos":          patch_pos,
        "sampling_n":         SAMPLING_N,
        "sampling_temp":      SAMPLING_TEMP,
        "max_new_tokens":     MAX_NEW_TOKENS,
        "clean_prompt":       pair["clean_prompt"],
        "corrupt_prompt":     pair["corrupt_prompt"],
        "clean_rhyme_word":   pair["clean_rhyme_word"],
        "corrupt_rhyme_word": pair["corrupt_rhyme_word"],
        "n_layers":           n_layers,
        "baseline": {
            "completions":                        baseline_completions,
            "unpatched_clean_clean_rhyme_rate":   baseline_clean_rate,
            "unpatched_clean_corrupt_rhyme_rate": baseline_corrupt_rate,
        },
        "results": layer_results,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"    Saved: {json_path}", flush=True)
    return export

# ── Main ───────────────────────────────────────────────────────────────────────

def run_all():
    model, tokenizer = load_model()
    d_model  = model.config.hidden_size
    n_layers = get_n_layers(model)
    device   = get_input_device(model)

    print(f"\nCaching donor prompt activations...", flush=True)
    example_prompt = PROMPT_PAIRS[0]["corrupt_prompt"]
    ref_patch_pos, _ = find_patch_pos(tokenizer, example_prompt, PATCH_OFFSET)
    donor_ids = tokenizer(DONOR_PROMPT, return_tensors="pt")["input_ids"]
    donor_len = donor_ids.shape[1]
    if ref_patch_pos >= donor_len:
        print(f"  WARNING: donor too short ({donor_len}). Using last token.")
        donor_cache_pos = donor_len - 1
    else:
        donor_cache_pos = ref_patch_pos
    donor_cache = cache_hidden_states_at_pos(model, tokenizer, DONOR_PROMPT, donor_cache_pos)
    print(f"  Donor cache ready at abs pos {donor_cache_pos}", flush=True)

    results_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", "LLAMA70B_BASELINE",
    )

    for pair in PROMPT_PAIRS:
        pair_id = pair["pair_id"]
        clean_prompt = pair["clean_prompt"]
        print(f"\n{'='*60}\nPair: {pair_id}\n{'='*60}", flush=True)

        patch_pos, tok_str = find_patch_pos(tokenizer, pair["corrupt_prompt"], PATCH_OFFSET)
        print(f"  Patch position: {patch_pos} (token: {repr(tok_str)})", flush=True)

        bl_completions  = sample_completions(model, tokenizer, clean_prompt, SAMPLING_N, SAMPLING_TEMP)
        bl_clean_rate   = rhyme_rate(bl_completions, clean_prompt, pair["clean_rhyme_word"])
        bl_corrupt_rate = rhyme_rate(bl_completions, clean_prompt, pair["corrupt_rhyme_word"])
        print(f"    clean_rate={bl_clean_rate:.3f}  corrupt_rate={bl_corrupt_rate:.3f}", flush=True)

        zero_vecs = [torch.zeros(1, d_model, device=device) for _ in range(n_layers)]
        donor_vecs = donor_cache

        for patch_mode, patch_vecs in [("zero_vector", zero_vecs), ("donor_prompt", donor_vecs)]:
            out_dir = os.path.join(results_root, pair_id, patch_mode)
            run_baseline(
                model, tokenizer, pair,
                patch_mode=patch_mode,
                patch_vecs=patch_vecs,
                patch_pos=patch_pos,
                baseline_completions=bl_completions,
                baseline_clean_rate=bl_clean_rate,
                baseline_corrupt_rate=bl_corrupt_rate,
                out_dir=out_dir,
            )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    run_all()
