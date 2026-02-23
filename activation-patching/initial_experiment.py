import json
import os
import torch
import nltk
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

# ── Config ─────────────────────────────────────────────────────────────────────

RUN_NAME = "better_extraction_qwen2.5_14b_newline_sampling"

MODEL_NAME = "Qwen/Qwen2.5-14B"

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
SAMPLING_N    = 100    # completions per layer
SAMPLING_TEMP = 0.7   # temperature for sampling

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
    """Extract the last alphabetic word from a generated completion."""
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
    Extract rhyme word from first generated line.
    For this prompt, that is the word before the 3rd newline in full text.
    """
    # Prompt already contains line breaks; next newline ends first generated line.
    target_newline_index = prompt.count("\n") + 1
    rhyme_word = word_before_nth_newline(full_text, target_newline_index)
    if rhyme_word:
        return rhyme_word
    # Fallback: if no newline was generated, use final generated word.
    if full_text.startswith(prompt):
        return last_word(full_text[len(prompt):])
    return last_word(full_text)

def sample_completions(model, prompt: str, n: int, temperature: float) -> list[str]:
    """Draw n sampled completions from the model given a prompt."""
    completions = []
    for _ in range(n):
        completion = model.generate(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
        )
        completions.append(completion)
    return completions

def rhyme_rate(completions: list[str], prompt: str, rhyme_set: set[str]) -> float:
    """Fraction of completions whose last word is in rhyme_set."""
    hits = sum(extract_rhyme_word(c, prompt) in rhyme_set for c in completions)
    return hits / len(completions) if completions else 0.0

# ── Model Loading ───────────────────────────────────────────────────────────────

def load_model():
    print(f"Loading {MODEL_NAME} via TransformerLens...")
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    model.eval()
    print(f"Loaded. Layers: {model.cfg.n_layers} | d_model: {model.cfg.d_model}")
    return model

# ── Main Experiment ─────────────────────────────────────────────────────────────

def run_experiment():
    model = load_model()

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
    clean_tokens   = model.to_tokens(CLEAN_PROMPT)
    corrupt_tokens = model.to_tokens(CORRUPT_PROMPT)

    if clean_tokens.shape[1] != corrupt_tokens.shape[1]:
        print(f"\nWARNING: token length mismatch ({clean_tokens.shape[1]} vs {corrupt_tokens.shape[1]})")
        print("Prompts should tokenize to the same length for clean positional alignment.")

    tok_list = corrupt_tokens[0].tolist()

    # --- Find patch position ---
    if PATCH_MODE == "newline":
        newline_id = model.to_tokens("\n", prepend_bos=False)[0, 0].item()
        newline_positions = [i for i, t in enumerate(tok_list) if t == newline_id]
        if newline_positions:
            patch_pos = max(newline_positions)
        else:
            last_newline_char = CORRUPT_PROMPT.rfind("\n")
            if last_newline_char == -1:
                raise ValueError("No newline character in corrupt prompt.")
            tokenizer = getattr(model, "tokenizer", None)
            if tokenizer is None:
                raise ValueError("No newline token found and model has no tokenizer for offset mapping.")
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
        corrupt_r1_ids = model.to_tokens(f" {CORRUPT_RHYME_WORD}", prepend_bos=False)[0].tolist()
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
    for i, tok in enumerate(corrupt_tokens[0]):
        marker = f" <-- patch target ({patch_label})" if i == patch_pos else ""
        print(f"  pos {i:2d}: {repr(model.to_string(tok.unsqueeze(0)))}{marker}")

    # --- Baseline completions ---
    print("\n── Baseline Completions (greedy) ──")
    clean_completion   = model.generate(CLEAN_PROMPT,   max_new_tokens=MAX_NEW_TOKENS, temperature=0)
    corrupt_completion = model.generate(CORRUPT_PROMPT, max_new_tokens=MAX_NEW_TOKENS, temperature=0)
    print(f"Clean   -> {repr(clean_completion)}")
    print(f"Corrupt -> {repr(corrupt_completion)}")
    clean_end   = extract_rhyme_word(clean_completion, CLEAN_PROMPT)
    corrupt_end = extract_rhyme_word(corrupt_completion, CORRUPT_PROMPT)
    print(f"\nClean ends with:   '{clean_end}' — rhymes with '{CLEAN_RHYME_WORD}'?   {clean_end in clean_rhymes}")
    print(f"Corrupt ends with: '{corrupt_end}' — rhymes with '{CORRUPT_RHYME_WORD}'? {corrupt_end in corrupt_rhymes}")

    # --- Sampling baseline (unpatched corrupt run) ---
    if SAMPLING_MODE:
        print(f"\n── Unpatched Corrupt Baseline ({SAMPLING_N} samples, T={SAMPLING_TEMP}) ──")
        baseline_samples = sample_completions(model, CORRUPT_PROMPT, SAMPLING_N, SAMPLING_TEMP)
        baseline_clean_rate   = rhyme_rate(baseline_samples, CORRUPT_PROMPT, clean_rhymes)
        baseline_corrupt_rate = rhyme_rate(baseline_samples, CORRUPT_PROMPT, corrupt_rhymes)
        print(f"  Rhymes with '{CLEAN_RHYME_WORD}' (clean): {baseline_clean_rate:.3f}")
        print(f"  Rhymes with '{CORRUPT_RHYME_WORD}' (corrupt): {baseline_corrupt_rate:.3f}")
    else:
        baseline_clean_rate   = None
        baseline_corrupt_rate = None

    # --- Cache clean activations ---
    print("\nCaching clean activations...")
    _, clean_cache = model.run_with_cache(CLEAN_PROMPT)

    # --- Sweep layers ---
    print(f"\nPatching at {patch_label} across all {model.cfg.n_layers} layers...")
    print(f"Mode: {'sampling (N=' + str(SAMPLING_N) + ', T=' + str(SAMPLING_TEMP) + ')' if SAMPLING_MODE else 'greedy'}\n")

    results = []

    for layer in range(model.cfg.n_layers):
        clean_vec = clean_cache[f"blocks.{layer}.hook_resid_pre"][:, patch_pos, :].clone()

        def patch_hook(value, hook, vec=clean_vec):
            if value.shape[1] > patch_pos:
                out = value.clone()
                out[:, patch_pos, :] = vec
                return out
            return value

        hook = (f"blocks.{layer}.hook_resid_pre", patch_hook)

        if SAMPLING_MODE:
            with model.hooks(fwd_hooks=[hook]):
                completions = sample_completions(model, CORRUPT_PROMPT, SAMPLING_N, SAMPLING_TEMP)
            clean_rate   = rhyme_rate(completions, CORRUPT_PROMPT, clean_rhymes)
            corrupt_rate = rhyme_rate(completions, CORRUPT_PROMPT, corrupt_rhymes)
            # Store all completions but only log summary
            result = {
                "layer": layer,
                "completions": completions,
                "clean_rhyme_rate":   clean_rate,
                "corrupt_rhyme_rate": corrupt_rate,
                "baseline_clean_rate":   baseline_clean_rate,
                "baseline_corrupt_rate": baseline_corrupt_rate,
            }
            delta = clean_rate - baseline_clean_rate
            print(f"  Layer {layer:2d}: clean_rhyme_rate={clean_rate:.3f} (baseline={baseline_clean_rate:.3f}, delta={delta:+.3f})")

        else:
            with model.hooks(fwd_hooks=[hook]):
                completion = model.generate(CORRUPT_PROMPT, max_new_tokens=MAX_NEW_TOKENS, temperature=0)
            end_word            = extract_rhyme_word(completion, CORRUPT_PROMPT)
            rhymes_with_clean   = end_word in clean_rhymes
            rhymes_with_corrupt = end_word in corrupt_rhymes
            result = {
                "layer": layer,
                "completion": completion,
                "end_word": end_word,
                "rhymes_with_clean": rhymes_with_clean,
                "rhymes_with_corrupt": rhymes_with_corrupt,
            }
            status = f"✓ rhymes with '{CLEAN_RHYME_WORD}'" if rhymes_with_clean else \
                     f"✗ rhymes with '{CORRUPT_RHYME_WORD}'" if rhymes_with_corrupt else \
                     f"? '{end_word}'"
            print(f"  Layer {layer:2d}: {status}  |  {repr(completion.strip())}")

        results.append(result)

    # --- Summary ---
    print(f"\n── Summary ──")
    if SAMPLING_MODE:
        best = max(results, key=lambda r: r["clean_rhyme_rate"])
        print(f"Best layer: {best['layer']} (clean_rhyme_rate={best['clean_rhyme_rate']:.3f}, baseline={baseline_clean_rate:.3f})")
    else:
        n_transferred = sum(r["rhymes_with_clean"] for r in results)
        print(f"Layers where patch transferred clean rhyme plan: {n_transferred} / {model.cfg.n_layers}")
        print(f"Successful layers: {[r['layer'] for r in results if r['rhymes_with_clean']]}")

    # --- Plot ---
    layers = [r["layer"] for r in results]
    fig, ax = plt.subplots(figsize=(14, 4))

    if SAMPLING_MODE:
        clean_rates = [r["clean_rhyme_rate"] for r in results]
        ax.bar(layers, clean_rates, color="steelblue", edgecolor="white", linewidth=0.5, label=f"'{CLEAN_RHYME_WORD}'-rhyme rate (patched)")
        ax.axhline(baseline_clean_rate, color="red", linestyle="--", linewidth=1.5, label=f"baseline corrupt rate ({baseline_clean_rate:.3f})")
        ax.set_ylabel(f"Fraction ending with '{CLEAN_RHYME_WORD}'-rhyme")
        ax.legend(loc="upper right")
    else:
        colors = [
            "steelblue" if r["rhymes_with_clean"]   else
            "salmon"    if r["rhymes_with_corrupt"]  else
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
    ax.set_xlim(-0.5, model.cfg.n_layers - 0.5)

    plt.tight_layout()
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    run_dir = os.path.join(results_dir, RUN_NAME)
    os.makedirs(run_dir, exist_ok=True)

    image_path = os.path.join(run_dir, f"patching_results_{PATCH_MODE}_{'sampling' if SAMPLING_MODE else 'greedy'}.png")
    plt.savefig(image_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {image_path}")

    # --- Save JSON ---
    export = {
        "run_name": RUN_NAME,
        "model_name": MODEL_NAME,
        "patch_mode": PATCH_MODE,
        "patch_label": patch_label,
        "patch_pos": int(patch_pos),
        "sampling_mode": SAMPLING_MODE,
        "sampling_n": SAMPLING_N if SAMPLING_MODE else None,
        "sampling_temp": SAMPLING_TEMP if SAMPLING_MODE else None,
        "max_new_tokens": MAX_NEW_TOKENS,
        "clean_prompt": CLEAN_PROMPT,
        "corrupt_prompt": CORRUPT_PROMPT,
        "clean_rhyme_word": CLEAN_RHYME_WORD,
        "corrupt_rhyme_word": CORRUPT_RHYME_WORD,
        "baseline": {
            "clean_completion": clean_completion,
            "corrupt_completion": corrupt_completion,
            "unpatched_corrupt_clean_rhyme_rate": baseline_clean_rate,
            "unpatched_corrupt_corrupt_rhyme_rate": baseline_corrupt_rate,
        },
        "n_layers": model.cfg.n_layers,
        "results": results,
    }
    json_path = os.path.join(run_dir, "generations.json")
    with open(json_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"Generations saved to {json_path}")

    return results


if __name__ == "__main__":
    results = run_experiment()