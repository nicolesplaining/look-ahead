import json
import os
import torch
import nltk
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

# ── Config ─────────────────────────────────────────────────────────────────────

# Name for this run: outputs go to activation-patching/results/<RUN_NAME>/
RUN_NAME = "qwen2.5_14b_newline"

MODEL_NAME = "Qwen/Qwen2.5-14B"

CLEAN_PROMPT   = "A rhyming couplet:\nHe felt a sudden urge to sleep,\n"
CORRUPT_PROMPT = "A rhyming couplet:\nHe felt a sudden urge to rest,\n"

CLEAN_RHYME_WORD   = "sleep"
CORRUPT_RHYME_WORD = "rest"

# "newline" → patch at the final newline token (i=0 in paper notation)
# "r1"      → patch at the r1 token itself ("sleep" / "rest")
PATCH_MODE = "newline"

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

    if PATCH_MODE not in ("newline", "r1"):
        raise ValueError(f"PATCH_MODE must be 'newline' or 'r1', got '{PATCH_MODE}'")

    # --- Tokenize ---
    clean_tokens   = model.to_tokens(CLEAN_PROMPT)
    corrupt_tokens = model.to_tokens(CORRUPT_PROMPT)

    if clean_tokens.shape[1] != corrupt_tokens.shape[1]:
        print(f"\nWARNING: token length mismatch ({clean_tokens.shape[1]} vs {corrupt_tokens.shape[1]})")
        print("Prompts should tokenize to the same length for clean positional alignment.")

    tok_list = corrupt_tokens[0].tolist()

    # --- Find patch position based on mode ---
    if PATCH_MODE == "newline":
        newline_id = model.to_tokens("\n", prepend_bos=False)[0, 0].item()
        newline_positions = [i for i, t in enumerate(tok_list) if t == newline_id]
        if newline_positions:
            patch_pos = max(newline_positions)
        else:
            # Tokenizer may not have a standalone newline token (e.g. Qwen2.5); find token covering the last "\n" by character offset
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
        # Find the corrupt r1 token ("rest") — search as " rest" (space-prefixed)
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
    print(f"\nCorrupt tokens:")
    for i, tok in enumerate(corrupt_tokens[0]):
        marker = f" <-- patch target ({patch_label})" if i == patch_pos else ""
        print(f"  pos {i:2d}: {repr(model.to_string(tok.unsqueeze(0)))}{marker}")

    # --- Baseline completions ---
    print("\n── Baseline Completions ──")
    clean_completion   = model.generate(CLEAN_PROMPT,   max_new_tokens=MAX_NEW_TOKENS, temperature=0)
    corrupt_completion = model.generate(CORRUPT_PROMPT, max_new_tokens=MAX_NEW_TOKENS, temperature=0)
    print(f"Clean   -> {repr(clean_completion)}")
    print(f"Corrupt -> {repr(corrupt_completion)}")

    clean_end   = last_word(clean_completion[len(CLEAN_PROMPT):])
    corrupt_end = last_word(corrupt_completion[len(CORRUPT_PROMPT):])
    print(f"\nClean ends with:   '{clean_end}' — rhymes with '{CLEAN_RHYME_WORD}'?   {clean_end in clean_rhymes}")
    print(f"Corrupt ends with: '{corrupt_end}' — rhymes with '{CORRUPT_RHYME_WORD}'? {corrupt_end in corrupt_rhymes}")

    # --- Cache clean activations ---
    print("\nCaching clean activations...")
    _, clean_cache = model.run_with_cache(CLEAN_PROMPT)

    # --- Sweep layers ---
    print(f"\nPatching at {patch_label} across all {model.cfg.n_layers} layers...")
    print("Running greedy completion for each layer...\n")

    results = []

    for layer in range(model.cfg.n_layers):
        clean_vec = clean_cache[f"blocks.{layer}.hook_resid_pre"][:, patch_pos, :].clone()

        def patch_hook(value, hook, vec=clean_vec):
            if value.shape[1] > patch_pos:
                out = value.clone()
                out[:, patch_pos, :] = vec
                return out
            return value

        with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", patch_hook)]):
            completion = model.generate(CORRUPT_PROMPT, max_new_tokens=MAX_NEW_TOKENS, temperature=0)

        end_word            = last_word(completion[len(CORRUPT_PROMPT):])
        rhymes_with_clean   = end_word in clean_rhymes
        rhymes_with_corrupt = end_word in corrupt_rhymes

        results.append({
            "layer": layer,
            "completion": completion,
            "end_word": end_word,
            "rhymes_with_clean": rhymes_with_clean,
            "rhymes_with_corrupt": rhymes_with_corrupt,
        })

        status = f"✓ rhymes with '{CLEAN_RHYME_WORD}'" if rhymes_with_clean else \
                 f"✗ rhymes with '{CORRUPT_RHYME_WORD}'" if rhymes_with_corrupt else \
                 f"? '{end_word}'"
        print(f"  Layer {layer:2d}: {status}  |  {repr(completion.strip())}")

    # --- Summary ---
    n_transferred = sum(r["rhymes_with_clean"] for r in results)
    print(f"\n── Summary ──")
    print(f"Layers where patch transferred clean rhyme plan: {n_transferred} / {model.cfg.n_layers}")
    print(f"Successful layers: {[r['layer'] for r in results if r['rhymes_with_clean']]}")

    # --- Plot ---
    layers = [r["layer"] for r in results]
    colors = [
        "steelblue" if r["rhymes_with_clean"]   else
        "salmon"    if r["rhymes_with_corrupt"]  else
        "lightgray"
        for r in results
    ]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(layers, [1] * len(layers), color=colors, edgecolor="white", linewidth=0.5)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="steelblue", label=f"Ends with '{CLEAN_RHYME_WORD}'-rhyme (transfer ✓)"),
        Patch(facecolor="salmon",    label=f"Ends with '{CORRUPT_RHYME_WORD}'-rhyme (no transfer)"),
        Patch(facecolor="lightgray", label="Neither"),
    ], loc="upper right")

    ax.set_xlabel(f"Layer (patch mode: {PATCH_MODE} @ {patch_label})")
    ax.set_yticks([])
    ax.set_xticks(layers)
    ax.set_title(
        f"Does patching [{patch_label}] transfer the rhyme plan? (mode={PATCH_MODE})\n"
        f"{MODEL_NAME} | clean r1='{CLEAN_RHYME_WORD}' → corrupt run (r1='{CORRUPT_RHYME_WORD}')"
    )
    ax.set_xlim(-0.5, model.cfg.n_layers - 0.5)

    plt.tight_layout()
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    run_dir = os.path.join(results_dir, RUN_NAME)
    os.makedirs(run_dir, exist_ok=True)

    # Save plot
    image_path = os.path.join(run_dir, f"patching_results_{PATCH_MODE}.png")
    plt.savefig(image_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {image_path}")

    # Save generations and run metadata to JSON
    export = {
        "run_name": RUN_NAME,
        "model_name": MODEL_NAME,
        "patch_mode": PATCH_MODE,
        "patch_label": patch_label,
        "patch_pos": int(patch_pos),
        "max_new_tokens": MAX_NEW_TOKENS,
        "clean_prompt": CLEAN_PROMPT,
        "corrupt_prompt": CORRUPT_PROMPT,
        "clean_rhyme_word": CLEAN_RHYME_WORD,
        "corrupt_rhyme_word": CORRUPT_RHYME_WORD,
        "baseline": {
            "clean_completion": clean_completion,
            "corrupt_completion": corrupt_completion,
        },
        "n_layers": model.cfg.n_layers,
        "results": [
            {
                "layer": r["layer"],
                "completion": r["completion"],
                "end_word": r["end_word"],
                "rhymes_with_clean": r["rhymes_with_clean"],
                "rhymes_with_corrupt": r["rhymes_with_corrupt"],
            }
            for r in results
        ],
    }
    json_path = os.path.join(run_dir, "generations.json")
    with open(json_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"Generations saved to {json_path}")

    return results


if __name__ == "__main__":
    results = run_experiment()