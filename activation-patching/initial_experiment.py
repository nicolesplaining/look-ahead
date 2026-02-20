import os
import torch
import nltk
import matplotlib.pyplot as plt

# TransformerLens imports BertForPreTraining/T5ForConditionalGeneration, removed in transformers 4.46+.
# Qwen3 requires transformers >= 4.51. Stub them so we can use Qwen3 (they're only used for BERT/T5).
import transformers as _tf
if not hasattr(_tf, "BertForPreTraining"):
    _tf.BertForPreTraining = type("BertForPreTraining", (), {})
if not hasattr(_tf, "T5ForConditionalGeneration"):
    _tf.T5ForConditionalGeneration = type("T5ForConditionalGeneration", (), {})

# Force eager load of Qwen2/Qwen3 so AutoModelForCausalLM finds them (avoids lazy-module lookup failure).
# Qwen2.5 uses the same qwen2 package in transformers 4.57. Requires Pillow>=9.1 (for PIL.Image.Resampling).
def _ensure_qwen_models_loaded():
    for full_mod_name, attr in [
        ("transformers.models.qwen2.modeling_qwen2", "Qwen2ForCausalLM"),
        ("transformers.models.qwen3.modeling_qwen3", "Qwen3ForCausalLM"),
    ]:
        try:
            mod = __import__(full_mod_name, fromlist=[attr])
            cls = getattr(mod, attr, None)
            if cls is not None:
                setattr(_tf, attr, cls)
                # Also inject into the subpackage so lookup in transformers.models.qwen2 succeeds.
                pkg_name = full_mod_name.rsplit(".", 1)[0]  # e.g. transformers.models.qwen2
                pkg = __import__(pkg_name, fromlist=[attr])
                setattr(pkg, attr, cls)
        except Exception:
            pass
_ensure_qwen_models_loaded()

from transformer_lens import HookedTransformer

# Qwen2.5-14B: same scale as Qwen3-14B, stable with TransformerLens + transformers 4.x/5.x
MODEL_NAME = "Qwen/Qwen2.5-14B"

CLEAN_PROMPT   = "A rhyming couplet:\nHe felt a sudden urge to sleep,\n"
CORRUPT_PROMPT = "A rhyming couplet:\nHe felt a sudden urge to rest,\n"

CLEAN_RHYME_WORD   = "sleep"
CORRUPT_RHYME_WORD = "rest"

MAX_NEW_TOKENS = 20

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

def run_experiment():
    model = load_model()
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

    clean_tokens   = model.to_tokens(CLEAN_PROMPT)
    corrupt_tokens = model.to_tokens(CORRUPT_PROMPT)

    if clean_tokens.shape[1] != corrupt_tokens.shape[1]:
        print(f"\nWARNING: token length mismatch ({clean_tokens.shape[1]} vs {corrupt_tokens.shape[1]})")
        print("Prompts should tokenize to the same length for clean positional alignment.")
    newline_id  = model.to_tokens("\n", prepend_bos=False)[0, 0].item()
    tok_list    = corrupt_tokens[0].tolist()
    newline_indices = [i for i, t in enumerate(tok_list) if t == newline_id]
    if not newline_indices:
        raise ValueError("No newline token found in corrupt prompt; cannot align patch position.")
    newline_pos = max(newline_indices)

    print(f"\nCorrupt tokens:")
    for i, tok in enumerate(corrupt_tokens[0]):
        marker = f" <-- newline (pos={newline_pos})" if i == newline_pos else ""
        print(f"  pos {i:2d}: {repr(model.to_string(tok.unsqueeze(0)))}{marker}")
    print("\n── Baseline Completions ──")
    clean_completion   = model.generate(CLEAN_PROMPT,   max_new_tokens=MAX_NEW_TOKENS, temperature=0)
    corrupt_completion = model.generate(CORRUPT_PROMPT, max_new_tokens=MAX_NEW_TOKENS, temperature=0)
    print(f"Clean   -> {repr(clean_completion)}")
    print(f"Corrupt -> {repr(corrupt_completion)}")

    clean_end   = last_word(clean_completion)
    corrupt_end = last_word(corrupt_completion)
    print(f"\nClean ends with:   '{clean_end}' — rhymes with '{CLEAN_RHYME_WORD}'?   {clean_end in clean_rhymes}")
    print(f"Corrupt ends with: '{corrupt_end}' — rhymes with '{CORRUPT_RHYME_WORD}'? {corrupt_end in corrupt_rhymes}")

    print("\nCaching clean activations...")
    _, clean_cache = model.run_with_cache(CLEAN_PROMPT)

    print(f"\nPatching clean newline (pos={newline_pos}) into corrupt run, sweeping all {model.cfg.n_layers} layers...")
    print("Running greedy completion for each layer...\n")

    results = []

    for layer in range(model.cfg.n_layers):
        clean_vec = clean_cache[f"blocks.{layer}.hook_resid_pre"][:, newline_pos, :].clone()

        def patch_hook(value, hook, vec=clean_vec):
            # Only patch on the prompt pass (seq has prompt length); generation steps have seq_len=1
            if value.shape[1] > newline_pos:
                out = value.clone()
                out[:, newline_pos, :] = vec
                return out
            return value

        with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", patch_hook)]):
            completion = model.generate(CORRUPT_PROMPT, max_new_tokens=MAX_NEW_TOKENS, temperature=0)

        end_word            = last_word(completion)
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

    n_transferred = sum(r["rhymes_with_clean"] for r in results)
    print(f"\n── Summary ──")
    print(f"Layers where patch transferred clean rhyme plan: {n_transferred} / {model.cfg.n_layers}")
    print(f"Successful layers: {[r['layer'] for r in results if r['rhymes_with_clean']]}")

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

    ax.set_xlabel("Layer patched at newline position")
    ax.set_yticks([])
    ax.set_xticks(layers)
    ax.set_title(
        f"Does patching the newline activation transfer the rhyme plan?\n"
        f"{MODEL_NAME} | clean r1='{CLEAN_RHYME_WORD}' → corrupt run (r1='{CORRUPT_RHYME_WORD}')"
    )
    ax.set_xlim(-0.5, model.cfg.n_layers - 0.5)

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "patching_results_qwen2_5_14b.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")

    return results


if __name__ == "__main__":
    results = run_experiment()