"""
All-layers activation patching for Llama-3.1-70B.

Mirrors patch_all_layers_unified.py but pinned to Llama-3.1-70B with 4-bit
loading (matching our per-layer 70B run). Sweeps 6 positions × 5 pairs;
patches all 80 layers simultaneously at the chosen position.

Output schema is identical to canonical, so plotting scripts work unchanged.
Saved to results/llama-3.1-70b-all-layers/<pair_id>/<pos_id>/generations.json
"""

import gc
import json
import os
from datetime import datetime, timezone

import pronouncing
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Config ────────────────────────────────────────────────────────────────────

RUN_NAME   = "llama-3.1-70b-all-layers"
MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"

SAMPLING_N     = 20
SAMPLING_TEMP  = 0.7
MAX_NEW_TOKENS = 20
BATCH_SIZE     = 20

POSITIONS = [
    {"offset": -2, "pos_id": "i_minus2"},
    {"offset": -1, "pos_id": "i_minus1"},
    {"offset":  0, "pos_id": "i_0"},
    {"offset": +1, "pos_id": "i_plus1"},
    {"offset": +2, "pos_id": "i_plus2"},
    {"offset": +3, "pos_id": "i_plus3"},
]

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

# ── Rhyme Helpers ──────────────────────────────────────────────────────────────

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
    print(f"Loading {MODEL_NAME} (4-bit)...")
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
    print(f"Loaded. Layers={model.config.num_hidden_layers}", flush=True)
    return model, tokenizer

def get_input_device(model): return model.model.embed_tokens.weight.device
def get_layers(model):       return model.model.layers

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

# ── Position Resolution ────────────────────────────────────────────────────────

def find_patch_pos(tokenizer, prompt, offset):
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    token_ids = enc["input_ids"]
    om = enc["offset_mapping"]
    nls = [i for i, ch in enumerate(prompt) if ch == "\n"]
    if len(nls) < 2:
        raise ValueError(f"Need 2 newlines, found {len(nls)}")
    second_nl = nls[1]
    nl_tok = next((i for i, (s, e) in enumerate(om) if s <= second_nl < e), None)
    if nl_tok is None:
        raise ValueError("Could not find newline token")
    patch_pos = nl_tok + offset
    if patch_pos < 0 or patch_pos >= len(token_ids):
        raise ValueError(f"offset={offset} out-of-bounds (n_tokens={len(token_ids)})")
    tok_str = tokenizer.decode([token_ids[patch_pos]])
    sign = "+" if offset >= 0 else ""
    label = f"i={sign}{offset} (pos={patch_pos}, tok={repr(tok_str)})"
    return patch_pos, label

# ── Caching ────────────────────────────────────────────────────────────────────

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
        raise RuntimeError("Failed to cache some layers")
    return cached

# ── Hook (all-layers) ──────────────────────────────────────────────────────────

def make_patch_hook(patch_vec, patch_pos):
    def hook(mod, args):
        h = args[0]
        if h.shape[1] <= 1 or h.shape[1] <= patch_pos:
            return args
        out = h.clone()
        out[:, patch_pos, :] = patch_vec.to(out.device, dtype=out.dtype)
        return (out,) + args[1:]
    return hook

# ── Position run (all layers patched simultaneously) ──────────────────────────

def run_position(model, tokenizer, pair, offset, pos_id,
                 baseline_completions, baseline_clean_rate, baseline_corrupt_rate,
                 pair_dir):
    n_layers = model.config.num_hidden_layers
    pos_dir   = os.path.join(pair_dir, pos_id)
    json_path = os.path.join(pos_dir, "generations.json")
    if os.path.exists(json_path):
        print(f"  Position {pos_id} — already done", flush=True)
        with open(json_path) as f:
            return json.load(f)

    clean_prompt   = pair["clean_prompt"]
    corrupt_prompt = pair["corrupt_prompt"]
    clean_rhyme_word   = pair["clean_rhyme_word"]
    corrupt_rhyme_word = pair["corrupt_rhyme_word"]

    try:
        cpos, clabel = find_patch_pos(tokenizer, corrupt_prompt, offset)
        kpos, klabel = find_patch_pos(tokenizer, clean_prompt,   offset)
    except ValueError as e:
        print(f"  Position {pos_id}: SKIP ({e})", flush=True)
        return None

    if cpos != kpos:
        print(f"  WARNING patch pos differs (corrupt={cpos}, clean={kpos})", flush=True)
    patch_pos = cpos

    print(f"  Position {pos_id} | patch at {clabel} | all {n_layers} layers", flush=True)
    corrupt_cache = cache_hidden_states_at_pos(model, tokenizer, corrupt_prompt, cpos)

    layers = get_layers(model)
    handles = []
    for i in range(n_layers):
        handles.append(layers[i].register_forward_pre_hook(
            make_patch_hook(corrupt_cache[i], patch_pos)
        ))
    try:
        completions = sample_completions(model, tokenizer, clean_prompt, SAMPLING_N, SAMPLING_TEMP)
    finally:
        for h in handles:
            h.remove()
        gc.collect(); torch.cuda.empty_cache()

    cr = rhyme_rate(completions, clean_prompt, clean_rhyme_word)
    cor = rhyme_rate(completions, clean_prompt, corrupt_rhyme_word)
    print(f"    corrupt_rhyme_rate={cor:.3f}  clean_rhyme_rate={cr:.3f}  "
          f"delta={cor - baseline_corrupt_rate:+.3f}", flush=True)

    export = {
        "timestamp_utc":       datetime.now(timezone.utc).isoformat(),
        "pair_id":             pair["pair_id"],
        "pos_id":              pos_id,
        "offset":              offset,
        "model_name":          MODEL_NAME,
        "patch_mode":          "all_layers_simultaneous",
        "patch_direction":     "corrupt->clean",
        "n_layers_patched":    n_layers,
        "corrupt_patch_pos":   cpos,
        "corrupt_patch_label": clabel,
        "clean_patch_pos":     kpos,
        "clean_patch_label":   klabel,
        "sampling_n":          SAMPLING_N,
        "sampling_temp":       SAMPLING_TEMP,
        "max_new_tokens":      MAX_NEW_TOKENS,
        "clean_prompt":        clean_prompt,
        "corrupt_prompt":      corrupt_prompt,
        "clean_rhyme_word":    clean_rhyme_word,
        "corrupt_rhyme_word":  corrupt_rhyme_word,
        "completions":         completions,
        "clean_rhyme_rate":    cr,
        "corrupt_rhyme_rate":  cor,
        "delta_corrupt_rate":  cor - baseline_corrupt_rate,
        "baseline": {
            "completions":        baseline_completions,
            "clean_rhyme_rate":   baseline_clean_rate,
            "corrupt_rhyme_rate": baseline_corrupt_rate,
        },
    }
    os.makedirs(pos_dir, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(export, f, indent=2)
    return export

# ── Pair ───────────────────────────────────────────────────────────────────────

def run_pair(model, tokenizer, pair, results_dir):
    pair_id  = pair["pair_id"]
    pair_dir = os.path.join(results_dir, pair_id)
    meta_path = os.path.join(pair_dir, "pair_meta.json")
    print(f"\n{'='*60}\nPair: {pair_id}\n{'='*60}", flush=True)

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            pm = json.load(f)
        baseline_completions  = pm.get("baseline_completions", [])
        baseline_clean_rate   = pm["baseline_clean_rate"]
        baseline_corrupt_rate = pm["baseline_corrupt_rate"]
        print(f"  Baseline loaded: clean={baseline_clean_rate:.3f} corrupt={baseline_corrupt_rate:.3f}", flush=True)
    else:
        print("  Greedy + sampled baseline...", flush=True)
        clean_g   = generate_text(model, tokenizer, pair["clean_prompt"],   0)
        corrupt_g = generate_text(model, tokenizer, pair["corrupt_prompt"], 0)
        baseline_completions = sample_completions(
            model, tokenizer, pair["clean_prompt"], SAMPLING_N, SAMPLING_TEMP
        )
        baseline_clean_rate   = rhyme_rate(baseline_completions, pair["clean_prompt"], pair["clean_rhyme_word"])
        baseline_corrupt_rate = rhyme_rate(baseline_completions, pair["clean_prompt"], pair["corrupt_rhyme_word"])
        print(f"    clean={baseline_clean_rate:.3f} corrupt={baseline_corrupt_rate:.3f}", flush=True)
        os.makedirs(pair_dir, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump({
                "pair_id": pair_id,
                "clean_prompt":   pair["clean_prompt"],
                "corrupt_prompt": pair["corrupt_prompt"],
                "clean_rhyme_word":   pair["clean_rhyme_word"],
                "corrupt_rhyme_word": pair["corrupt_rhyme_word"],
                "clean_completion":   clean_g,
                "corrupt_completion": corrupt_g,
                "baseline_completions":  baseline_completions,
                "baseline_clean_rate":   baseline_clean_rate,
                "baseline_corrupt_rate": baseline_corrupt_rate,
            }, f, indent=2)

    by_pos = {}
    for pos in POSITIONS:
        by_pos[pos["pos_id"]] = run_position(
            model, tokenizer, pair, pos["offset"], pos["pos_id"],
            baseline_completions, baseline_clean_rate, baseline_corrupt_rate, pair_dir,
        )
    return by_pos

# ── Main ───────────────────────────────────────────────────────────────────────

def run_experiment():
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "results", RUN_NAME)
    os.makedirs(results_dir, exist_ok=True)
    model, tokenizer = load_model()
    for pair in PROMPT_PAIRS:
        run_pair(model, tokenizer, pair, results_dir)
    print(f"\nDone. Results in {results_dir}", flush=True)

if __name__ == "__main__":
    run_experiment()
