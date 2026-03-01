import os
import re
import urllib.request
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Shared Config (model identity + rhyme evaluation) ──────────────────────────
# Generation parameters (N_SAMPLES, TEMPERATURE, etc.) and prompts
# (NEUTRAL_PROMPT, sweep settings) live in the individual experiment scripts.

MODEL_NAME = "Qwen/Qwen3-32B"
DEVICE     = "cuda"

# Steering vector construction: v = h_clean - h_corrupt at the newline token.
# CLEAN  ends with "sleep" → primes "-eep" rhyme family
# CORRUPT ends with "rest"  → primes "-est" rhyme family
CLEAN_PROMPT   = "A rhyming couplet:\nShe closed her eyes to get some sleep,\n"
CORRUPT_PROMPT = "A rhyming couplet:\nShe closed her eyes to get some rest,\n"

CMU_DICT_URL         = "http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b"
CMU_DICT_PATH        = "cmudict-0.7b"
CLEAN_RHYME_SUFFIX   = "IY1 P"    # -eep: sleep, deep, keep, weep, steep, leap, creep
CORRUPT_RHYME_SUFFIX = "EH1 S T"  # -est: rest, best, chest, nest, west, test, quest

RESULTS_DIR = "steering_results"
FIGURES_DIR = "figures"

# ── Model ───────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer():
    """Returns (model, tokenizer, num_layers, hidden_dim)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"Loaded {MODEL_NAME}: {num_layers} layers, hidden_dim={hidden_dim}")
    return model, tokenizer, num_layers, hidden_dim

# ── CMU Dict & Rhyme Checking ───────────────────────────────────────────────────

def load_cmu_dict():
    """Downloads CMU dict if not cached. Returns {WORD: [phoneme_string, ...]}."""
    if not os.path.exists(CMU_DICT_PATH):
        print(f"Downloading CMU dict to {CMU_DICT_PATH}...")
        urllib.request.urlretrieve(CMU_DICT_URL, CMU_DICT_PATH)
    cmu = {}
    with open(CMU_DICT_PATH, encoding="latin-1") as f:
        for line in f:
            if line.startswith(";;;"):
                continue
            parts = line.strip().split("  ", 1)
            if len(parts) != 2:
                continue
            word   = re.sub(r"\(\d+\)$", "", parts[0])
            phones = parts[1]
            cmu.setdefault(word, []).append(phones)
    return cmu


def get_rhyme_suffix(phoneme_string):
    """Everything from the last primary-stressed vowel onward. E.g. "S L IY1 P" -> "IY1 P"."""
    phones = phoneme_string.split()
    for i in range(len(phones) - 1, -1, -1):
        if phones[i].endswith("1"):
            return " ".join(phones[i:])
    return phoneme_string


def build_rhyme_checkers(cmu_dict):
    """Returns (rhymes_with_eep, rhymes_with_est), each word -> bool."""
    def make_checker(target_suffix):
        def checker(word):
            word = word.upper().strip(".,!?;:'\"")
            if word not in cmu_dict:
                return False
            return any(get_rhyme_suffix(p) == target_suffix for p in cmu_dict[word])
        return checker
    return make_checker(CLEAN_RHYME_SUFFIX), make_checker(CORRUPT_RHYME_SUFFIX)

# ── Steering Vector ─────────────────────────────────────────────────────────────

def extract_steering_vector(model, tokenizer, num_layers):
    """
    v_ell = h_clean_ell - h_corrupt_ell at the last token (newline) of each prompt.
    Returns {layer_index: tensor(hidden_dim,)} on CPU, float32.
    """
    states = {}
    with torch.no_grad():
        for prompt, label in [(CLEAN_PROMPT, "clean"), (CORRUPT_PROMPT, "corrupt")]:
            inputs  = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            # hidden_states[ell+1] = output of layer ell; shape (1, seq_len, hidden_dim)
            states[label] = {
                ell: outputs.hidden_states[ell + 1][0, -1, :].float().cpu()
                for ell in range(num_layers)
            }
    return {ell: states["clean"][ell] - states["corrupt"][ell] for ell in range(num_layers)}

# ── Hook ────────────────────────────────────────────────────────────────────────

def make_steering_hook(steering_vector, alpha, target_seq_len):
    """
    Forward hook that adds alpha * steering_vector to the last token of the prefill pass.
    No-op on decode steps (seq_len == 1). Fires at most once.
    Returns (hook_fn, hook_fired) where hook_fired[0] is set True after firing.
    """
    hook_fired = [False]
    vec = (alpha * steering_vector).to(DEVICE)

    def hook_fn(module, input, output):
        hidden_states = output[0]
        if hidden_states.shape[1] != target_seq_len or hook_fired[0]:
            return output
        hidden_states = hidden_states.clone()
        hidden_states[:, -1, :] += vec.to(hidden_states.dtype)
        hook_fired[0] = True
        return (hidden_states,) + output[1:]

    return hook_fn, hook_fired

# ── Generation ──────────────────────────────────────────────────────────────────

def run_steered_generation(model, tokenizer, prompt, layer, alpha, steering_vectors,
                           n_samples, temperature, max_new_tokens, top_p):
    """Generate n_samples completions of prompt steered at `layer` with `alpha`."""
    inputs     = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    prompt_len = inputs["input_ids"].shape[1]
    completions = []

    for _ in range(n_samples):
        hook_fn, hook_fired = make_steering_hook(
            steering_vectors[layer], alpha, target_seq_len=prompt_len
        )
        handle = model.model.layers[layer].register_forward_hook(hook_fn)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        handle.remove()
        assert hook_fired[0], f"Hook did not fire for layer {layer}."
        completions.append(
            tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
        )

    return completions


def run_baseline_generation(model, tokenizer, prompt, n_samples, temperature,
                            max_new_tokens, top_p):
    """Generate n_samples unsteered completions of prompt."""
    inputs     = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    prompt_len = inputs["input_ids"].shape[1]
    completions = []

    for _ in range(n_samples):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        completions.append(
            tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
        )

    return completions

# ── Evaluation ──────────────────────────────────────────────────────────────────

def evaluate_completions(completions, rhymes_with_eep, rhymes_with_est):
    """
    Returns (clean_rate, corrupt_rate, neither_rate, clean_words, corrupt_words).
    clean_rate  = proportion of completions containing any "-eep" word
    corrupt_rate = proportion containing any "-est" word
    """
    clean_hits, corrupt_hits = [], []
    clean_words_found, corrupt_words_found = [], []

    for completion in completions:
        words = completion.replace("\n", " ").split()
        found_clean = found_corrupt = False
        for word in words:
            if not found_clean and rhymes_with_eep(word):
                clean_words_found.append(word.lower().strip(".,!?;:'\""))
                found_clean = True
            if not found_corrupt and rhymes_with_est(word):
                corrupt_words_found.append(word.lower().strip(".,!?;:'\""))
                found_corrupt = True
        clean_hits.append(found_clean)
        corrupt_hits.append(found_corrupt)

    n = len(completions)
    clean_rate   = sum(clean_hits) / n if n else 0.0
    corrupt_rate = sum(corrupt_hits) / n if n else 0.0
    neither_rate = sum(not c and not r for c, r in zip(clean_hits, corrupt_hits)) / n if n else 0.0
    return clean_rate, corrupt_rate, neither_rate, clean_words_found, corrupt_words_found
