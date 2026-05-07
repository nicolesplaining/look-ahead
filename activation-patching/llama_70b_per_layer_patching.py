"""
Per-layer, per-position activation patching for Llama-3.1-70B.
For each layer and each target position (last word token, newline token),
patches the corrupt hidden state into the clean forward pass and measures
corrupt rhyme rate averaged over 20 prompt pairs.
"""

import gc
import json
import os
import torch
import pronouncing
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────

RUN_NAME   = "llama-3.1-70b-per-layer-per-position"
MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"

SAMPLING_N    = 10   # samples per prompt pair per (layer, position)
SAMPLING_TEMP = 0.7
MAX_NEW_TOKENS = 12

# 20 prompt pairs: (clean_first_line, corrupt_first_line, clean_rhyme, corrupt_rhyme)
# Each pair produces prompts: "A rhyming couplet:\n{line},\n"
# Token lengths must match — all lines end in a single rhyme word before ",\n"
PROMPT_PAIRS = [
    ("She felt a sudden sense of fright",  "She felt a sudden sense of fear",   "fright", "fear"),
    ("The sky was filled with silent doom", "The sky was filled with silent dread","doom",  "dread"),
    ("The children laughed in bliss",      "The children laughed in joy",        "bliss",  "joy"),
    ("She wandered home into the dark",    "She wandered home into the night",   "dark",   "night"),
    ("I never knew the depth of grief",    "I never knew the depth of pain",     "grief",  "pain"),
    ("The candle burned so very bright",   "The candle burned so very dim",      "bright", "dim"),
    ("The morning came without a sound",   "The morning came without a trace",   "sound",  "trace"),
    ("He gazed upon the silver moon",      "He gazed upon the glowing sun",      "moon",   "sun"),
    ("She sang a soft and gentle song",    "She sang a soft and tender hymn",    "song",   "hymn"),
    ("The rose bloomed red against the vine","The rose bloomed pale against the wall","vine","wall"),
    ("They walked beneath the fading light","They walked beneath the falling rain","light", "rain"),
    ("The river ran so cold and deep",     "The river ran so cold and still",    "deep",   "still"),
    ("He stood alone upon the hill",       "He stood alone beside the stone",    "hill",   "stone"),
    ("The storm arrived without a sign",   "The storm arrived without a word",   "sign",   "word"),
    ("She held the letter in her hand",    "She held the letter in her palm",    "hand",   "palm"),
    ("The clock struck twelve without a chime","The clock struck twelve without a bell","chime","bell"),
    ("He smiled and said a fond farewell", "He smiled and said a long goodbye",  "farewell","goodbye"),
    ("The old dog slept beside the gate",  "The old dog slept beside the door",  "gate",   "door"),
    ("She closed her eyes and felt the breeze","She closed her eyes and felt the wind","breeze","wind"),
    ("The stars appeared above the town",  "The stars appeared above the field", "town",   "field"),
]

POSITIONS = [-1, 0]   # -1 = last word token (Llama: "fright"), 0 = newline token (",\n")

# ── Rhyme Helpers ───────────────────────────────────────────────────────────────

def _rhyme_score(w1: str, w2: str) -> Optional[bool]:
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

def extract_rhyme_word(full_text: str, prompt: str) -> str:
    target_nl = prompt.count("\n") + 1
    nl_pos = [i for i, ch in enumerate(full_text) if ch == "\n"]
    if len(nl_pos) >= target_nl:
        end   = nl_pos[target_nl - 1]
        start = nl_pos[target_nl - 2] + 1 if target_nl >= 2 else 0
        w = last_word(full_text[start:end])
        if w:
            return w
    if full_text.startswith(prompt):
        return last_word(full_text[len(prompt):])
    return last_word(full_text)

def rhyme_rate(completions: list, prompt: str, rhyme_word: str) -> float:
    hits = sum(1 for c in completions
               if _rhyme_score(extract_rhyme_word(c, prompt), rhyme_word) is True)
    return hits / len(completions) if completions else 0.0

# ── Model ───────────────────────────────────────────────────────────────────────

def load_model():
    print(f"Loading {MODEL_NAME} (4-bit quantized)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        gpu_used_gib = torch.cuda.memory_allocated() / (1024**3)
        print(f"GPU memory after load: {gpu_used_gib:.2f} GiB")
    print(f"Loaded. Layers={model.config.num_hidden_layers} d_model={model.config.hidden_size}")
    return model, tokenizer

def get_input_device(model):
    return model.model.embed_tokens.weight.device

def generate_text(model, tokenizer, prompt: str, temperature: float) -> str:
    device = get_input_device(model)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
        )
    result = tokenizer.decode(out[0], skip_special_tokens=True)
    del out, enc
    gc.collect()
    torch.cuda.empty_cache()
    return result

def cache_hidden_states(model, tokenizer, prompt: str) -> tuple:
    device = get_input_device(model)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**enc, output_hidden_states=True)
    return tuple(h.detach() for h in outputs.hidden_states)

# ── Position Resolution ─────────────────────────────────────────────────────────

def resolve_position(prompt: str, tokenizer, offset: int) -> int:
    """Return absolute token index for position offset relative to second newline."""
    nl_chars = [i for i, ch in enumerate(prompt) if ch == "\n"]
    if len(nl_chars) < 2:
        raise ValueError(f"Need 2 newlines, found {len(nl_chars)}")
    second_nl = nl_chars[1]
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    om = enc["offset_mapping"]
    nl_tok = next((i for i, (s, e) in enumerate(om) if s <= second_nl < e), None)
    if nl_tok is None:
        raise ValueError("Could not locate second newline token")
    return nl_tok + offset

# ── Hook ────────────────────────────────────────────────────────────────────────

def make_patch_hook(patch_vec: torch.Tensor, patch_pos: int):
    def hook_fn(module, args):
        h = args[0]
        if h.shape[1] <= 1:   # decode step
            return args
        if h.shape[1] > patch_pos:
            out = h.clone()
            out[:, patch_pos, :] = patch_vec.to(h.device)
            return (out,) + args[1:]
        return args
    return hook_fn

# ── Main ────────────────────────────────────────────────────────────────────────

def build_prompt(line: str) -> str:
    return f"A rhyming couplet:\n{line},\n"

def run_experiment():
    model, tokenizer = load_model()
    n_layers = model.config.num_hidden_layers

    results_by_position = {pos: [] for pos in POSITIONS}

    for layer in tqdm(range(n_layers), desc="Layers"):
        layer_pos_results = {}

        for pos_offset in POSITIONS:
            corrupt_rates = []

            for clean_line, corrupt_line, clean_rhyme, corrupt_rhyme in PROMPT_PAIRS:
                clean_prompt   = build_prompt(clean_line)
                corrupt_prompt = build_prompt(corrupt_line)

                # resolve absolute token position
                try:
                    abs_pos = resolve_position(clean_prompt, tokenizer, pos_offset)
                except Exception as e:
                    print(f"  WARNING: could not resolve pos {pos_offset} for pair '{clean_rhyme}/{corrupt_rhyme}': {e}")
                    continue

                # cache corrupt hidden states
                corrupt_hs = cache_hidden_states(model, tokenizer, corrupt_prompt)
                patch_vec  = corrupt_hs[layer][:, abs_pos, :].clone()
                del corrupt_hs
                gc.collect()
                torch.cuda.empty_cache()

                # patch and sample
                handle = model.model.layers[layer].register_forward_pre_hook(
                    make_patch_hook(patch_vec, abs_pos)
                )
                try:
                    completions = [
                        generate_text(model, tokenizer, clean_prompt, SAMPLING_TEMP)
                        for _ in range(SAMPLING_N)
                    ]
                    cr = rhyme_rate(completions, clean_prompt, corrupt_rhyme)
                finally:
                    handle.remove()
                    del patch_vec
                    gc.collect()
                    torch.cuda.empty_cache()

                corrupt_rates.append(cr)

            mean_cr = sum(corrupt_rates) / len(corrupt_rates) if corrupt_rates else 0.0
            layer_pos_results[pos_offset] = mean_cr
            print(f"  Layer {layer:3d}  pos={pos_offset:+d}  corrupt_rhyme_rate={mean_cr:.3f}")

        for pos_offset in POSITIONS:
            results_by_position[pos_offset].append({
                "layer": layer,
                "corrupt_rhyme_rate": layer_pos_results.get(pos_offset, 0.0),
            })

    # ── Save ────────────────────────────────────────────────────────────────────
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", RUN_NAME)
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "generations.json")
    with open(json_path, "w") as f:
        json.dump({
            "run_name":    RUN_NAME,
            "model_name":  MODEL_NAME,
            "n_layers":    n_layers,
            "positions":   POSITIONS,
            "sampling_n":  SAMPLING_N,
            "sampling_temp": SAMPLING_TEMP,
            "n_prompt_pairs": len(PROMPT_PAIRS),
            "results_by_position": {str(k): v for k, v in results_by_position.items()},
        }, f, indent=2)
    print(f"\nSaved to {json_path}")


if __name__ == "__main__":
    run_experiment()
