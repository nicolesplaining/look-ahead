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

SAMPLING_N    = 50    # completions per layer
SAMPLING_TEMP = 0.7   # temperature for sampling

MAX_NEW_TOKENS = 16

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
    """Extract the last alphabetic word from a generated completion."""
    words = [w.strip(".,!?\"' ") for w in text.split()]
    words = [w for w in words if w.isalpha()]
    return words[-1].lower() if words else ""

# ── Model Loading ───────────────────────────────────────────────────────────────

def load_model():
    print(f"Loading {MODEL_NAME} via transformers...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    d_model  = model.config.hidden_size
    print(f"Loaded. Layers: {n_layers} | d_model: {d_model}")
    return model, tokenizer

# ── Generation Helpers ──────────────────────────────────────────────────────────

def get_input_device(model) -> torch.device:
    """Return the device of the embedding layer (where input_ids should be sent)."""
    return model.model.embed_tokens.weight.device

def generate_text(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float = 0.0) -> str:
    """Generate text from prompt; returns full string (prompt + generation)."""
    device = get_input_device(model)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def sample_completions(model, tokenizer, prompt: str, n: int, temperature: float) -> list[str]:
    """Draw n sampled completions from the model given a prompt."""
    return [generate_text(model, tokenizer, prompt, MAX_NEW_TOKENS, temperature) for _ in range(n)]

def rhyme_rate(completions: list[str], prompt: str, rhyme_set: set[str]) -> float:
    """Fraction of completions whose last word is in rhyme_set."""
    generated = [c[len(prompt):] for c in completions]
    hits = sum(last_word(g) in rhyme_set for g in generated)
    return hits / len(completions) if completions else 0.0

# ── Activation Caching ──────────────────────────────────────────────────────────

def cache_hidden_states(model, tokenizer, prompt: str) -> tuple:
    """
    Run a single forward pass with output_hidden_states=True.

    Returns a tuple of (n_layers + 1) tensors, each [1, seq_len, d_model]:
      hidden_states[0]     = embedding output        = resid_pre for layer 0
      hidden_states[L]     = output of layer (L-1)   = resid_pre for layer L

    This maps exactly to TransformerLens's blocks.{L}.hook_resid_pre.
    """
    device = get_input_device(model)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    return tuple(h.detach() for h in outputs.hidden_states)

# ── Activation Patching ─────────────────────────────────────────────────────────

def make_patch_hook(clean_vec: torch.Tensor, patch_pos: int):
    """
    Returns a forward_pre_hook for model.model.layers[L].

    Replaces hidden_states[:, patch_pos, :] with clean_vec before the layer runs.
    No-op when seq_len == 1 (autoregressive decode steps), so KV-cached generation
    is not affected after the prefill patch.

    clean_vec: [1, d_model] tensor (device doesn't matter; moved on use)
    """
    def hook_fn(module, args):
        # args[0] is hidden_states: [batch, seq_len, d_model]
        hidden_states = args[0]
        if hidden_states.shape[1] > patch_pos:
            patched = hidden_states.clone()
            patched[:, patch_pos, :] = clean_vec.to(hidden_states.device)
            return (patched,) + args[1:]
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

    # --- Tokenize ---
    clean_ids   = tokenizer(CLEAN_PROMPT,   return_tensors="pt").input_ids
    corrupt_ids = tokenizer(CORRUPT_PROMPT, return_tensors="pt").input_ids
    if clean_ids.shape[1] != corrupt_ids.shape[1]:
        print(f"\nWARNING: token length mismatch ({clean_ids.shape[1]} vs {corrupt_ids.shape[1]})")
        print("Prompts should tokenize to the same length for correct positional alignment.")

    tok_list = clean_ids[0].tolist()

    # --- Find patch position (last newline in the clean/target prompt) ---
    newline_id = tokenizer("\n", add_special_tokens=False).input_ids[0]
    newline_positions = [i for i, t in enumerate(tok_list) if t == newline_id]
    if not newline_positions:
        raise ValueError("No newline token found in clean prompt.")
    patch_pos = max(newline_positions)
    patch_label = f"newline (pos={patch_pos})"

    print(f"\nPatch direction: corrupt → clean (injecting '{CORRUPT_RHYME_WORD}' activations into '{CLEAN_RHYME_WORD}' run)")
    print(f"Patch position: {patch_label} | N={SAMPLING_N} T={SAMPLING_TEMP}")
    print(f"\nClean prompt tokens:")
    for i, tok_id in enumerate(tok_list):
        marker = " <-- patch target" if i == patch_pos else ""
        print(f"  pos {i:2d}: {repr(tokenizer.decode([tok_id]))}{marker}")

    # --- Baseline: unpatched clean prompt ---
    print(f"\n── Unpatched Clean Baseline ({SAMPLING_N} samples, T={SAMPLING_TEMP}) ──")
    baseline_samples      = sample_completions(model, tokenizer, CLEAN_PROMPT, SAMPLING_N, SAMPLING_TEMP)
    baseline_clean_rate   = rhyme_rate(baseline_samples, CLEAN_PROMPT, clean_rhymes)
    baseline_corrupt_rate = rhyme_rate(baseline_samples, CLEAN_PROMPT, corrupt_rhymes)
    print(f"  Rhymes with '{CLEAN_RHYME_WORD}' (expected high): {baseline_clean_rate:.3f}")
    print(f"  Rhymes with '{CORRUPT_RHYME_WORD}' (expected low): {baseline_corrupt_rate:.3f}")

    # --- Cache corrupt activations (the source we inject from) ---
    print("\nCaching corrupt activations...")
    corrupt_hidden_states = cache_hidden_states(model, tokenizer, CORRUPT_PROMPT)

    # --- Sweep layers ---
    print(f"\nPatching {patch_label} across all {n_layers} layers (corrupt→clean)...\n")

    results = []

    for layer in range(n_layers):
        # Grab the corrupt prompt's hidden state at the patch position for this layer
        corrupt_vec = corrupt_hidden_states[layer][:, patch_pos, :].clone()

        handle = model.model.layers[layer].register_forward_pre_hook(
            make_patch_hook(corrupt_vec, patch_pos)
        )
        try:
            completions   = sample_completions(model, tokenizer, CLEAN_PROMPT, SAMPLING_N, SAMPLING_TEMP)
            clean_rate    = rhyme_rate(completions, CLEAN_PROMPT, clean_rhymes)
            corrupt_rate  = rhyme_rate(completions, CLEAN_PROMPT, corrupt_rhymes)
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
    best = max(results, key=lambda r: r["corrupt_rhyme_rate"])
    print(f"\nBest layer: {best['layer']} (corrupt_rhyme_rate={best['corrupt_rhyme_rate']:.3f}, baseline={baseline_corrupt_rate:.3f})")

    # --- Plot ---
    layers       = [r["layer"] for r in results]
    corrupt_rates = [r["corrupt_rhyme_rate"] for r in results]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(layers, corrupt_rates, color="salmon", edgecolor="white", linewidth=0.5,
           label=f"'{CORRUPT_RHYME_WORD}'-rhyme rate (patched clean run)")
    ax.axhline(baseline_corrupt_rate, color="red", linestyle="--", linewidth=1.5,
               label=f"baseline clean rate ({baseline_corrupt_rate:.3f})")
    ax.set_xlabel(f"Layer ({patch_label})")
    ax.set_ylabel(f"Fraction ending with '{CORRUPT_RHYME_WORD}'-rhyme")
    ax.set_xticks(layers)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_title(
        f"Corrupt → clean patching @ {patch_label}: does '{CORRUPT_RHYME_WORD}' steer the '{CLEAN_RHYME_WORD}' run?\n"
        f"{MODEL_NAME} | N={SAMPLING_N} T={SAMPLING_TEMP}"
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
    export = {
        "run_name":    RUN_NAME,
        "model_name":  MODEL_NAME,
        "patch_direction": "corrupt→clean",
        "patch_pos":   int(patch_pos),
        "sampling_n":    SAMPLING_N,
        "sampling_temp": SAMPLING_TEMP,
        "max_new_tokens": MAX_NEW_TOKENS,
        "clean_prompt":   CLEAN_PROMPT,
        "corrupt_prompt": CORRUPT_PROMPT,
        "clean_rhyme_word":   CLEAN_RHYME_WORD,
        "corrupt_rhyme_word": CORRUPT_RHYME_WORD,
        "baseline_clean_rate":   baseline_clean_rate,
        "baseline_corrupt_rate": baseline_corrupt_rate,
        "n_layers": n_layers,
        "results":  results,
    }
    json_path = os.path.join(run_dir, "generations.json")
    with open(json_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"Generations saved to {json_path}")

    return results


if __name__ == "__main__":
    results = run_experiment()
