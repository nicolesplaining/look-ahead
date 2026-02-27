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

# "newline" → patch at the final newline token (i=0 in paper notation)
# "r1"      → patch at the r1 token itself ("sleep" / "rest")
PATCH_MODE = "newline"

# False → greedy (one completion per layer, fast)
# True  → sample N completions per layer, report rhyme rate
SAMPLING_MODE = True
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

    # --- Validate flags ---
    if PATCH_MODE not in ("newline", "r1"):
        raise ValueError(f"PATCH_MODE must be 'newline' or 'r1', got '{PATCH_MODE}'")

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
        print("Prompts should tokenize to the same length for clean positional alignment.")

    tok_list = corrupt_ids[0].tolist()

    # --- Find patch position ---
    if PATCH_MODE == "newline":
        newline_id = tokenizer("\n", add_special_tokens=False).input_ids[0]
        newline_positions = [i for i, t in enumerate(tok_list) if t == newline_id]
        if newline_positions:
            patch_pos = max(newline_positions)
        else:
            last_newline_char = CORRUPT_PROMPT.rfind("\n")
            if last_newline_char == -1:
                raise ValueError("No newline character in corrupt prompt.")
            enc = tokenizer(CORRUPT_PROMPT, return_offsets_mapping=True, add_special_tokens=True)
            offset_mapping = enc.get("offset_mapping") or []
            patch_pos = None
            for i, (start, end) in enumerate(offset_mapping):
                if start <= last_newline_char < end:
                    patch_pos = i
                    break
            if patch_pos is None:
                raise ValueError("Could not find token covering newline in corrupt prompt.")
        patch_label = f"newline (i=0, pos={patch_pos})"

    elif PATCH_MODE == "r1":
        corrupt_r1_ids = tokenizer(f" {CORRUPT_RHYME_WORD}", add_special_tokens=False).input_ids
        patch_pos = None
        for i in range(len(tok_list) - len(corrupt_r1_ids), -1, -1):
            if tok_list[i:i + len(corrupt_r1_ids)] == corrupt_r1_ids:
                patch_pos = i
                break
        if patch_pos is None:
            raise ValueError(f"Could not find '{CORRUPT_RHYME_WORD}' token in corrupt prompt.")
        patch_label = f"r1 token ('{CORRUPT_RHYME_WORD}', pos={patch_pos})"

    print(f"\nPatch mode: {PATCH_MODE} → patching at {patch_label}")
    print(f"Sampling mode: {SAMPLING_MODE}" + (f" (N={SAMPLING_N}, T={SAMPLING_TEMP})" if SAMPLING_MODE else " (greedy)"))
    print(f"\nCorrupt tokens:")
    for i, tok_id in enumerate(tok_list):
        tok_str = tokenizer.decode([tok_id])
        marker  = f" <-- patch target ({patch_label})" if i == patch_pos else ""
        print(f"  pos {i:2d}: {repr(tok_str)}{marker}")

    # --- Baseline completions ---
    print("\n── Baseline Completions (greedy) ──")
    clean_completion   = generate_text(model, tokenizer, CLEAN_PROMPT,   MAX_NEW_TOKENS, temperature=0)
    corrupt_completion = generate_text(model, tokenizer, CORRUPT_PROMPT, MAX_NEW_TOKENS, temperature=0)
    print(f"Clean   -> {repr(clean_completion)}")
    print(f"Corrupt -> {repr(corrupt_completion)}")
    clean_end   = last_word(clean_completion[len(CLEAN_PROMPT):])
    corrupt_end = last_word(corrupt_completion[len(CORRUPT_PROMPT):])
    print(f"\nClean ends with:   '{clean_end}' — rhymes with '{CLEAN_RHYME_WORD}'?   {clean_end in clean_rhymes}")
    print(f"Corrupt ends with: '{corrupt_end}' — rhymes with '{CORRUPT_RHYME_WORD}'? {corrupt_end in corrupt_rhymes}")

    # --- Sampling baseline (unpatched corrupt run) ---
    if SAMPLING_MODE:
        print(f"\n── Unpatched Corrupt Baseline ({SAMPLING_N} samples, T={SAMPLING_TEMP}) ──")
        baseline_samples      = sample_completions(model, tokenizer, CORRUPT_PROMPT, SAMPLING_N, SAMPLING_TEMP)
        baseline_clean_rate   = rhyme_rate(baseline_samples, CORRUPT_PROMPT, clean_rhymes)
        baseline_corrupt_rate = rhyme_rate(baseline_samples, CORRUPT_PROMPT, corrupt_rhymes)
        print(f"  Rhymes with '{CLEAN_RHYME_WORD}' (clean): {baseline_clean_rate:.3f}")
        print(f"  Rhymes with '{CORRUPT_RHYME_WORD}' (corrupt): {baseline_corrupt_rate:.3f}")
    else:
        baseline_clean_rate   = None
        baseline_corrupt_rate = None

    # --- Cache clean activations ---
    # hidden_states[L] == resid_pre for layer L (same indexing as TL's hook_resid_pre)
    print("\nCaching clean activations...")
    clean_hidden_states = cache_hidden_states(model, tokenizer, CLEAN_PROMPT)

    # --- Sweep layers ---
    print(f"\nPatching at {patch_label} across all {n_layers} layers...")
    print(f"Mode: {'sampling (N=' + str(SAMPLING_N) + ', T=' + str(SAMPLING_TEMP) + ')' if SAMPLING_MODE else 'greedy'}\n")

    results = []

    for layer in range(n_layers):
        # clean_hidden_states[layer] shape: [1, seq_len, d_model]
        # Slice the patch position → [1, d_model]
        clean_vec = clean_hidden_states[layer][:, patch_pos, :].clone()

        hook_fn = make_patch_hook(clean_vec, patch_pos)
        handle  = model.model.layers[layer].register_forward_pre_hook(hook_fn)

        try:
            if SAMPLING_MODE:
                completions  = sample_completions(model, tokenizer, CORRUPT_PROMPT, SAMPLING_N, SAMPLING_TEMP)
                clean_rate   = rhyme_rate(completions, CORRUPT_PROMPT, clean_rhymes)
                corrupt_rate = rhyme_rate(completions, CORRUPT_PROMPT, corrupt_rhymes)
                result = {
                    "layer": layer,
                    "completions": completions,
                    "clean_rhyme_rate":    clean_rate,
                    "corrupt_rhyme_rate":  corrupt_rate,
                    "baseline_clean_rate":   baseline_clean_rate,
                    "baseline_corrupt_rate": baseline_corrupt_rate,
                }
                delta = clean_rate - baseline_clean_rate
                print(f"  Layer {layer:2d}: clean_rhyme_rate={clean_rate:.3f} (baseline={baseline_clean_rate:.3f}, delta={delta:+.3f})")
            else:
                completion          = generate_text(model, tokenizer, CORRUPT_PROMPT, MAX_NEW_TOKENS, temperature=0)
                end_word            = last_word(completion[len(CORRUPT_PROMPT):])
                rhymes_with_clean   = end_word in clean_rhymes
                rhymes_with_corrupt = end_word in corrupt_rhymes
                result = {
                    "layer": layer,
                    "completion": completion,
                    "end_word": end_word,
                    "rhymes_with_clean":   rhymes_with_clean,
                    "rhymes_with_corrupt": rhymes_with_corrupt,
                }
                status = f"✓ rhymes with '{CLEAN_RHYME_WORD}'" if rhymes_with_clean else \
                         f"✗ rhymes with '{CORRUPT_RHYME_WORD}'" if rhymes_with_corrupt else \
                         f"? '{end_word}'"
                print(f"  Layer {layer:2d}: {status}  |  {repr(completion.strip())}")
        finally:
            handle.remove()

        results.append(result)

    # --- Summary ---
    print(f"\n── Summary ──")
    if SAMPLING_MODE:
        best = max(results, key=lambda r: r["clean_rhyme_rate"])
        print(f"Best layer: {best['layer']} (clean_rhyme_rate={best['clean_rhyme_rate']:.3f}, baseline={baseline_clean_rate:.3f})")
    else:
        n_transferred = sum(r["rhymes_with_clean"] for r in results)
        print(f"Layers where patch transferred clean rhyme plan: {n_transferred} / {n_layers}")
        print(f"Successful layers: {[r['layer'] for r in results if r['rhymes_with_clean']]}")

    # --- Plot ---
    layers = [r["layer"] for r in results]
    fig, ax = plt.subplots(figsize=(14, 4))

    if SAMPLING_MODE:
        clean_rates = [r["clean_rhyme_rate"] for r in results]
        ax.bar(layers, clean_rates, color="steelblue", edgecolor="white", linewidth=0.5,
               label=f"'{CLEAN_RHYME_WORD}'-rhyme rate (patched)")
        ax.axhline(baseline_clean_rate, color="red", linestyle="--", linewidth=1.5,
                   label=f"baseline corrupt rate ({baseline_clean_rate:.3f})")
        ax.set_ylabel(f"Fraction ending with '{CLEAN_RHYME_WORD}'-rhyme")
        ax.legend(loc="upper right")
    else:
        colors = [
            "steelblue" if r["rhymes_with_clean"]  else
            "salmon"    if r["rhymes_with_corrupt"] else
            "lightgray"
            for r in results
        ]
        ax.bar(layers, [1] * len(layers), color=colors, edgecolor="white", linewidth=0.5)
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor="steelblue", label=f"Ends with '{CLEAN_RHYME_WORD}'-rhyme (transfer ✓)"),
            Patch(facecolor="salmon",    label=f"Ends with '{CORRUPT_RHYME_WORD}'-rhyme (no transfer)"),
            Patch(facecolor="lightgray", label="Neither"),
        ], loc="upper right")
        ax.set_yticks([])

    ax.set_xlabel(f"Layer (patch mode: {PATCH_MODE} @ {patch_label})")
    ax.set_xticks(layers)
    ax.set_title(
        f"Does patching [{patch_label}] transfer the rhyme plan? "
        f"({'sampling N=' + str(SAMPLING_N) + ' T=' + str(SAMPLING_TEMP) if SAMPLING_MODE else 'greedy'})\n"
        f"{MODEL_NAME} | clean r1='{CLEAN_RHYME_WORD}' → corrupt run (r1='{CORRUPT_RHYME_WORD}')"
    )
    ax.set_xlim(-0.5, n_layers - 0.5)

    plt.tight_layout()
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    run_dir     = os.path.join(results_dir, RUN_NAME)
    os.makedirs(run_dir, exist_ok=True)

    image_path = os.path.join(run_dir, f"patching_results_{PATCH_MODE}_{'sampling' if SAMPLING_MODE else 'greedy'}.png")
    plt.savefig(image_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {image_path}")

    # --- Save JSON ---
    export = {
        "run_name":    RUN_NAME,
        "model_name":  MODEL_NAME,
        "patch_mode":  PATCH_MODE,
        "patch_label": patch_label,
        "patch_pos":   int(patch_pos),
        "sampling_mode": SAMPLING_MODE,
        "sampling_n":    SAMPLING_N if SAMPLING_MODE else None,
        "sampling_temp": SAMPLING_TEMP if SAMPLING_MODE else None,
        "max_new_tokens": MAX_NEW_TOKENS,
        "clean_prompt":   CLEAN_PROMPT,
        "corrupt_prompt": CORRUPT_PROMPT,
        "clean_rhyme_word":   CLEAN_RHYME_WORD,
        "corrupt_rhyme_word": CORRUPT_RHYME_WORD,
        "baseline": {
            "clean_completion":   clean_completion,
            "corrupt_completion": corrupt_completion,
            "unpatched_corrupt_clean_rhyme_rate":   baseline_clean_rate,
            "unpatched_corrupt_corrupt_rhyme_rate": baseline_corrupt_rate,
        },
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
