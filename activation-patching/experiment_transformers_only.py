import json
import os
import torch
import nltk
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ─────────────────────────────────────────────────────────────────────

RUN_NAME = "qwen3-32b"

MODEL_NAME = "Qwen/Qwen3-32B"

CLEAN_PROMPT   = "A rhyming couplet:\nHe felt a sudden urge to sleep,\n"
CORRUPT_PROMPT = "A rhyming couplet:\nHe felt a sudden urge to rest,\n"

CLEAN_RHYME_WORD   = "sleep"
CORRUPT_RHYME_WORD = "rest"

SAMPLING_N    = 50
SAMPLING_TEMP = 0.7

MAX_NEW_TOKENS = 20

# ── CMU Rhyme Lookup ───────────────────────────────────────────────────────────

def get_rhyme_tail(phones: list[str]) -> tuple:
    """Return phones from the last stressed vowel onward (defines the rhyme)."""
    for i in reversed(range(len(phones))):
        if phones[i][-1] in "12":
            return tuple(phones[i:])
    return tuple(phones[-2:])

def build_rhyme_set(word: str) -> set[str]:
    """Return all words in CMU dict that rhyme with `word`."""
    try:
        entries = nltk.corpus.cmudict.entries()
    except LookupError:
        nltk.download("cmudict")
        entries = nltk.corpus.cmudict.entries()

    cmu = {}
    for w, phones in entries:
        cmu.setdefault(w, []).append(phones)

    target_phones = cmu.get(word.lower())
    if not target_phones:
        raise ValueError(f"'{word}' not found in CMU dict")

    target_tail = get_rhyme_tail(target_phones[0])
    return {
        w for w, phones_list in cmu.items()
        if w != word.lower()
        and any(get_rhyme_tail(p) == target_tail for p in phones_list)
    }

def last_word(text: str) -> str:
    """Extract the last alphabetic word from a string."""
    words = [w.strip(".,!?\"' ") for w in text.split()]
    words = [w for w in words if w.isalpha()]
    return words[-1].lower() if words else ""

def word_before_nth_newline(text: str, n: int) -> str:
    """Return the last alphabetic word before the n-th newline in text."""
    if n <= 0:
        return ""
    newline_positions = [i for i, ch in enumerate(text) if ch == "\n"]
    if len(newline_positions) < n:
        return ""
    return last_word(text[:newline_positions[n - 1]])

def extract_rhyme_word(full_text: str, prompt: str) -> str:
    """
    Extract the rhyme word from the first generated line.
    Looks for the word before the (prompt_newlines + 1)-th newline in the full text.
    Fallback: last word of generated text if no newline was produced.
    """
    target_newline_index = prompt.count("\n") + 1
    rhyme_word = word_before_nth_newline(full_text, target_newline_index)
    if rhyme_word:
        return rhyme_word
    if full_text.startswith(prompt):
        return last_word(full_text[len(prompt):])
    return last_word(full_text)

def rhyme_rate(completions: list[str], prompt: str, rhyme_set: set[str]) -> float:
    """Fraction of completions whose rhyme word is in rhyme_set."""
    hits = sum(extract_rhyme_word(c, prompt) in rhyme_set for c in completions)
    return hits / len(completions) if completions else 0.0

# ── Model Loading ───────────────────────────────────────────────────────────────

def load_model():
    print(f"Loading {MODEL_NAME} via transformers...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Loaded. Layers: {model.config.num_hidden_layers} | d_model: {model.config.hidden_size}")
    return model, tokenizer

# ── Generation Helpers ──────────────────────────────────────────────────────────

def get_input_device(model) -> torch.device:
    return model.model.embed_tokens.weight.device

def generate_text(model, tokenizer, prompt: str, temperature: float) -> str:
    """Generate text from prompt; returns full string (prompt + generation)."""
    device = get_input_device(model)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def sample_completions(model, tokenizer, prompt: str, n: int, temperature: float) -> list[str]:
    """Draw n sampled completions from the model given a prompt."""
    return [generate_text(model, tokenizer, prompt, temperature) for _ in range(n)]

# ── Activation Caching & Patching ──────────────────────────────────────────────

def cache_hidden_states(model, tokenizer, prompt: str) -> tuple:
    """
    Run a forward pass with output_hidden_states=True.
    Returns tuple of (n_layers + 1) tensors [1, seq_len, d_model]:
      hidden_states[L] = resid_pre for layer L  (maps to TL's blocks.{L}.hook_resid_pre)
    """
    device = get_input_device(model)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**enc, output_hidden_states=True)
    return tuple(h.detach() for h in outputs.hidden_states)

def make_patch_hook(patch_vec: torch.Tensor, patch_pos: int):
    """
    forward_pre_hook for model.model.layers[L].
    Replaces hidden_states[:, patch_pos, :] with patch_vec at prefill;
    no-op during decode steps (seq_len == 1).
    """
    def hook_fn(module, args):
        h = args[0]
        if h.shape[1] > patch_pos:
            out = h.clone()
            out[:, patch_pos, :] = patch_vec.to(h.device)
            return (out,) + args[1:]
        return args
    return hook_fn

# ── Main Experiment ─────────────────────────────────────────────────────────────

def run_experiment():
    model, tokenizer = load_model()
    n_layers = model.config.num_hidden_layers

    # --- Build rhyme sets ---
    print(f"\nBuilding rhyme sets from CMU dict...")
    clean_rhymes   = build_rhyme_set(CLEAN_RHYME_WORD)
    corrupt_rhymes = build_rhyme_set(CORRUPT_RHYME_WORD)
    overlap = clean_rhymes & corrupt_rhymes
    if overlap:
        print(f"  Removing {len(overlap)} overlapping words from both sets")
        clean_rhymes   -= overlap
        corrupt_rhymes -= overlap
    print(f"  '{CLEAN_RHYME_WORD}' rhymes: {len(clean_rhymes)} words, e.g. {list(clean_rhymes)[:6]}")
    print(f"  '{CORRUPT_RHYME_WORD}' rhymes: {len(corrupt_rhymes)} words, e.g. {list(corrupt_rhymes)[:6]}")

    # --- Tokenize & find patch position ---
    clean_ids   = tokenizer(CLEAN_PROMPT,   return_tensors="pt").input_ids
    corrupt_ids = tokenizer(CORRUPT_PROMPT, return_tensors="pt").input_ids
    if clean_ids.shape[1] != corrupt_ids.shape[1]:
        print(f"\nWARNING: token length mismatch ({clean_ids.shape[1]} vs {corrupt_ids.shape[1]})")
        print("Prompts should tokenize to the same length for correct positional alignment.")

    # Find the token covering the last '\n' in the clean prompt via offset mapping
    last_newline_char = CLEAN_PROMPT.rfind("\n")
    enc = tokenizer(CLEAN_PROMPT, return_offsets_mapping=True, add_special_tokens=True)
    patch_pos = next(
        (i for i, (s, e) in enumerate(enc["offset_mapping"]) if s <= last_newline_char < e),
        None,
    )
    if patch_pos is None:
        raise ValueError("Could not find token covering last newline in clean prompt.")
    patch_label = f"newline (pos={patch_pos})"

    print(f"\nPatch direction: corrupt → clean")
    print(f"  Injecting '{CORRUPT_RHYME_WORD}' activations into '{CLEAN_RHYME_WORD}' run at {patch_label}")
    print(f"  N={SAMPLING_N}, T={SAMPLING_TEMP}")

    print(f"\nClean prompt tokens:")
    for i, tok_id in enumerate(clean_ids[0].tolist()):
        marker = " <-- patch target" if i == patch_pos else ""
        print(f"  pos {i:2d}: {repr(tokenizer.decode([tok_id]))}{marker}")

    # --- Greedy baselines ---
    print("\n── Baseline Completions (greedy) ──")
    clean_completion   = generate_text(model, tokenizer, CLEAN_PROMPT,   temperature=0)
    corrupt_completion = generate_text(model, tokenizer, CORRUPT_PROMPT, temperature=0)
    print(f"Clean   -> {repr(clean_completion)}")
    print(f"Corrupt -> {repr(corrupt_completion)}")
    clean_end   = extract_rhyme_word(clean_completion,   CLEAN_PROMPT)
    corrupt_end = extract_rhyme_word(corrupt_completion, CORRUPT_PROMPT)
    print(f"\nClean ends with:   '{clean_end}' — rhymes with '{CLEAN_RHYME_WORD}'?   {clean_end in clean_rhymes}")
    print(f"Corrupt ends with: '{corrupt_end}' — rhymes with '{CORRUPT_RHYME_WORD}'? {corrupt_end in corrupt_rhymes}")

    # --- Sampling baseline: unpatched clean run ---
    print(f"\n── Unpatched Clean Baseline ({SAMPLING_N} samples, T={SAMPLING_TEMP}) ──")
    baseline_samples      = sample_completions(model, tokenizer, CLEAN_PROMPT, SAMPLING_N, SAMPLING_TEMP)
    baseline_clean_rate   = rhyme_rate(baseline_samples, CLEAN_PROMPT, clean_rhymes)
    baseline_corrupt_rate = rhyme_rate(baseline_samples, CLEAN_PROMPT, corrupt_rhymes)
    print(f"  Rhymes with '{CLEAN_RHYME_WORD}' (expected high): {baseline_clean_rate:.3f}")
    print(f"  Rhymes with '{CORRUPT_RHYME_WORD}' (expected low): {baseline_corrupt_rate:.3f}")

    # --- Cache corrupt activations (source of patch) ---
    print("\nCaching corrupt activations...")
    corrupt_hs = cache_hidden_states(model, tokenizer, CORRUPT_PROMPT)

    # --- Layer sweep ---
    print(f"\nPatching {patch_label} across all {n_layers} layers (corrupt→clean, N={SAMPLING_N}, T={SAMPLING_TEMP})...\n")

    results = []

    for layer in range(n_layers):
        corrupt_vec = corrupt_hs[layer][:, patch_pos, :].clone()
        handle = model.model.layers[layer].register_forward_pre_hook(
            make_patch_hook(corrupt_vec, patch_pos)
        )
        try:
            completions  = sample_completions(model, tokenizer, CLEAN_PROMPT, SAMPLING_N, SAMPLING_TEMP)
            clean_rate   = rhyme_rate(completions, CLEAN_PROMPT, clean_rhymes)
            corrupt_rate = rhyme_rate(completions, CLEAN_PROMPT, corrupt_rhymes)
        finally:
            handle.remove()

        delta = corrupt_rate - baseline_corrupt_rate
        print(f"  Layer {layer:2d}: corrupt_rhyme_rate={corrupt_rate:.3f} (baseline={baseline_corrupt_rate:.3f}, delta={delta:+.3f})")
        results.append({
            "layer": layer,
            "completions": completions,
            "clean_rhyme_rate":      clean_rate,
            "corrupt_rhyme_rate":    corrupt_rate,
            "baseline_clean_rate":   baseline_clean_rate,
            "baseline_corrupt_rate": baseline_corrupt_rate,
        })

    # --- Summary ---
    print(f"\n── Summary ──")
    best = max(results, key=lambda r: r["corrupt_rhyme_rate"])
    print(f"Best layer: {best['layer']} (corrupt_rhyme_rate={best['corrupt_rhyme_rate']:.3f}, baseline={baseline_corrupt_rate:.3f})")

    # --- Plot ---
    layers       = [r["layer"] for r in results]
    corrupt_rates = [r["corrupt_rhyme_rate"] for r in results]
    clean_rates   = [r["clean_rhyme_rate"]   for r in results]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(layers, corrupt_rates, color="salmon", edgecolor="white", linewidth=0.5,
           label=f"'{CORRUPT_RHYME_WORD}'-rhyme rate (patched)")
    ax.plot(layers, clean_rates, color="steelblue", marker="o", markersize=3, linewidth=1.0,
            label=f"'{CLEAN_RHYME_WORD}'-rhyme rate (patched)")
    ax.axhline(baseline_corrupt_rate, color="red", linestyle="--", linewidth=1.5,
               label=f"baseline corrupt rate ({baseline_corrupt_rate:.3f})")
    ax.axhline(baseline_clean_rate, color="orange", linestyle="--", linewidth=1.5,
               label=f"baseline clean rate ({baseline_clean_rate:.3f})")
    ax.set_xlabel(f"Layer (patch: {patch_label})")
    ax.set_ylabel("Rhyme rate")
    ax.set_xticks(layers)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_title(
        f"Does patching [{patch_label}] transfer the rhyme plan? "
        f"(sampling N={SAMPLING_N} T={SAMPLING_TEMP})\n"
        f"{MODEL_NAME} | corrupt r1='{CORRUPT_RHYME_WORD}' → clean run (r1='{CLEAN_RHYME_WORD}')"
    )
    ax.legend(loc="upper right")
    plt.tight_layout()

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    run_dir     = os.path.join(results_dir, RUN_NAME)
    os.makedirs(run_dir, exist_ok=True)

    image_path = os.path.join(run_dir, "patching_results.png")
    plt.savefig(image_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {image_path}")

    # --- Save JSON ---
    json_path = os.path.join(run_dir, "generations.json")
    with open(json_path, "w") as f:
        json.dump({
            "run_name":    RUN_NAME,
            "model_name":  MODEL_NAME,
            "patch_direction": "corrupt→clean",
            "patch_label": patch_label,
            "patch_pos":   int(patch_pos),
            "sampling_n":    SAMPLING_N,
            "sampling_temp": SAMPLING_TEMP,
            "max_new_tokens": MAX_NEW_TOKENS,
            "clean_prompt":   CLEAN_PROMPT,
            "corrupt_prompt": CORRUPT_PROMPT,
            "clean_rhyme_word":   CLEAN_RHYME_WORD,
            "corrupt_rhyme_word": CORRUPT_RHYME_WORD,
            "baseline": {
                "clean_completion":   clean_completion,
                "corrupt_completion": corrupt_completion,
                "unpatched_clean_clean_rhyme_rate":   baseline_clean_rate,
                "unpatched_clean_corrupt_rhyme_rate": baseline_corrupt_rate,
            },
            "n_layers": n_layers,
            "results":  results,
        }, f, indent=2)
    print(f"Generations saved to {json_path}")

    return results


if __name__ == "__main__":
    results = run_experiment()
