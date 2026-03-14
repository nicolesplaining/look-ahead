"""
Activation-patching baseline experiments for Gemma-3-27B-IT.

Two baseline patch modes (neither uses corrupt activations):
  1. zero_vector  — replace hidden state at patch position with zeros
  2. donor_prompt — replace with hidden state from an unrelated prompt at the same
                    absolute token position

Key position: i=-2 (last word token for Gemma3 due to tokenization).
Sweeps all layers, 4 prompt pairs, N=20 samples each.

Results saved to results/GEMMA3_BASELINE/
"""

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone

import pronouncing
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# -- Config ----------------------------------------------------------------------

MODEL_NAME     = "google/gemma-3-27b-it"
SAMPLING_N     = 20
SAMPLING_TEMP  = 0.7
MAX_NEW_TOKENS = 20

PATCH_OFFSET = -2   # i=-2 is the last word token for Gemma3 due to tokenization

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
]

# -- Rhyme helpers ---------------------------------------------------------------

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
        c = w.strip(".,!?\"'—;: ")
        if c.isalpha():
            return c.lower()
    return ""

def word_before_nth_newline(text, n):
    if n <= 0:
        return ""
    nls = [i for i, ch in enumerate(text) if ch == "\n"]
    if len(nls) < n:
        return ""
    end   = nls[n - 1]
    start = nls[n - 2] + 1 if n >= 2 else 0
    return last_word(text[start:end])

def extract_rhyme_word(full_text, prompt):
    idx = prompt.count("\n") + 1
    w = word_before_nth_newline(full_text, idx)
    if w:
        return w
    if full_text.startswith(prompt):
        return last_word(full_text[len(prompt):])
    return last_word(full_text)

def rhyme_rate(completions, prompt, rhyme_word):
    hits = sum(
        1 for c in completions
        if _rhyme_score(extract_rhyme_word(c, prompt), rhyme_word) is True
    )
    return hits / len(completions) if completions else 0.0

# -- Model -----------------------------------------------------------------------

def load_model():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    from transformers import Gemma3ForConditionalGeneration
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    text_cfg = model.config.text_config
    print(f"Loaded. Layers: {text_cfg.num_hidden_layers} | d_model: {text_cfg.hidden_size}")
    return model, tokenizer

def get_input_device(model):
    return model.model.language_model.embed_tokens.weight.device

def get_layers(model):
    return model.model.language_model.layers

def get_n_layers(model):
    return model.config.text_config.num_hidden_layers

def get_d_model(model):
    return model.config.text_config.hidden_size

# -- Generation ------------------------------------------------------------------

def generate_text(model, tokenizer, prompt, temperature):
    device = get_input_device(model)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def sample_completions(model, tokenizer, prompt, n, temperature):
    return [
        generate_text(model, tokenizer, prompt, temperature)
        for _ in tqdm(range(n), desc="Sampling", leave=False)
    ]

# -- Position finding ------------------------------------------------------------

def find_patch_pos(tokenizer, prompt, offset):
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    token_ids      = enc["input_ids"]
    offset_mapping = enc["offset_mapping"]
    nls = [i for i, ch in enumerate(prompt) if ch == "\n"]
    if len(nls) < 2:
        raise ValueError("Need at least 2 newlines in prompt.")
    second_nl_char = nls[1]
    second_nl_tok = next(
        (i for i, (s, e) in enumerate(offset_mapping) if s <= second_nl_char < e),
        None,
    )
    if second_nl_tok is None:
        raise ValueError("Could not find token for second newline.")
    patch_pos = second_nl_tok + offset
    tok_str = tokenizer.decode([token_ids[patch_pos]])
    return patch_pos, tok_str

# -- Activation caching ----------------------------------------------------------

def cache_hidden_states_at_pos(model, tokenizer, prompt, patch_pos):
    layers = get_layers(model)
    device = get_input_device(model)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    cached = [None] * len(layers)
    handles = []

    def make_capture(idx):
        def hook(module, args):
            h = args[0]
            if h.shape[1] > patch_pos:
                cached[idx] = h[:, patch_pos, :].detach().clone()
        return hook

    for idx, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(make_capture(idx)))
    try:
        with torch.no_grad():
            model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    missing = [i for i, v in enumerate(cached) if v is None]
    if missing:
        raise RuntimeError(f"Failed to cache layers: {missing}")
    return cached

# -- Patching --------------------------------------------------------------------

@contextmanager
def patch_layer_at_pos(model, layer_idx, patch_pos, patch_vec):
    layer = get_layers(model)[layer_idx]

    def hook(module, args):
        h = args[0]
        if h.shape[1] <= patch_pos:
            return args
        out = h.clone()
        out[:, patch_pos, :] = patch_vec.to(out.device, dtype=out.dtype)
        return (out,) + args[1:]

    handle = layer.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()

# -- Single baseline run ---------------------------------------------------------

def run_baseline(model, tokenizer, pair, patch_mode, patch_vecs,
                 patch_pos, baseline_completions, baseline_clean_rate,
                 baseline_corrupt_rate, out_dir):
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

    best = max(layer_results, key=lambda r: r["corrupt_rhyme_rate"])
    print(f"    [{patch_mode}] best layer {best['layer']}: corrupt_rate={best['corrupt_rhyme_rate']:.3f} (baseline={baseline_corrupt_rate:.3f})")

    export = {
        "timestamp_utc":       datetime.now(timezone.utc).isoformat(),
        "model_name":          MODEL_NAME,
        "pair_id":             pair["pair_id"],
        "patch_mode":          patch_mode,
        "patch_offset":        PATCH_OFFSET,
        "patch_pos":           patch_pos,
        "sampling_n":          SAMPLING_N,
        "sampling_temp":       SAMPLING_TEMP,
        "max_new_tokens":      MAX_NEW_TOKENS,
        "clean_prompt":        pair["clean_prompt"],
        "corrupt_prompt":      pair["corrupt_prompt"],
        "clean_rhyme_word":    pair["clean_rhyme_word"],
        "corrupt_rhyme_word":  pair["corrupt_rhyme_word"],
        "n_layers":            n_layers,
        "baseline": {
            "completions":                          baseline_completions,
            "unpatched_clean_clean_rhyme_rate":     baseline_clean_rate,
            "unpatched_clean_corrupt_rhyme_rate":   baseline_corrupt_rate,
        },
        "results": layer_results,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "generations.json")
    with open(path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"    Saved: {path}")
    return export

# -- Main ------------------------------------------------------------------------

def run_all():
    model, tokenizer = load_model()
    d_model  = get_d_model(model)
    n_layers = get_n_layers(model)
    device   = get_input_device(model)

    # Pre-cache donor activations
    print(f"\nCaching donor prompt activations...")
    example_prompt = PROMPT_PAIRS[0]["corrupt_prompt"]
    ref_patch_pos, _ = find_patch_pos(tokenizer, example_prompt, PATCH_OFFSET)
    donor_ids = tokenizer(DONOR_PROMPT, return_tensors="pt")["input_ids"]
    donor_len = donor_ids.shape[1]
    if ref_patch_pos >= donor_len:
        print(f"  WARNING: donor prompt too short ({donor_len} tokens) for ref_patch_pos={ref_patch_pos}. Using last token.")
        donor_cache_pos = donor_len - 1
    else:
        donor_cache_pos = ref_patch_pos
    donor_cache = cache_hidden_states_at_pos(model, tokenizer, DONOR_PROMPT, donor_cache_pos)
    print(f"  Donor cache ready at absolute pos {donor_cache_pos} (token: {repr(tokenizer.decode([tokenizer(DONOR_PROMPT)['input_ids'][donor_cache_pos]]))})")

    results_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", "GEMMA3_BASELINE",
    )

    for pair in PROMPT_PAIRS:
        pair_id      = pair["pair_id"]
        clean_prompt = pair["clean_prompt"]
        print(f"\n{'='*60}\nPair: {pair_id}\n{'='*60}")

        patch_pos, tok_str = find_patch_pos(tokenizer, pair["corrupt_prompt"], PATCH_OFFSET)
        print(f"  Patch position: {patch_pos} (token: {repr(tok_str)})")

        # Unpatched baseline
        print(f"  Unpatched baseline (N={SAMPLING_N})...")
        bl_completions  = sample_completions(model, tokenizer, clean_prompt, SAMPLING_N, SAMPLING_TEMP)
        bl_clean_rate   = rhyme_rate(bl_completions, clean_prompt, pair["clean_rhyme_word"])
        bl_corrupt_rate = rhyme_rate(bl_completions, clean_prompt, pair["corrupt_rhyme_word"])
        print(f"    clean_rate={bl_clean_rate:.3f}  corrupt_rate={bl_corrupt_rate:.3f}")

        # Zero-vector patch vecs
        zero_vecs = [torch.zeros(1, d_model, device=device) for _ in range(n_layers)]

        # Donor patch vecs
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

    print("\nDone.")


if __name__ == "__main__":
    run_all()
