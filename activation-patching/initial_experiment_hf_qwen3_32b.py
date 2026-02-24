import json
import os
from contextlib import contextmanager

import matplotlib.pyplot as plt
import nltk
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ─────────────────────────────────────────────────────────────────────

RUN_NAME = "qwen3_32b_newline_sampling_hf"

MODEL_NAME = "Qwen/Qwen3-32B"

CLEAN_PROMPT = "A rhyming couplet:\nHe felt a sudden urge to sleep,\n"
CORRUPT_PROMPT = "A rhyming couplet:\nHe felt a sudden urge to rest,\n"

CLEAN_RHYME_WORD = "sleep"
CORRUPT_RHYME_WORD = "rest"

# "newline" → patch at the final newline token (i=0 in paper notation)
# "r1"      → patch at the r1 token itself ("sleep" / "rest")
PATCH_MODE = "newline"

# False → greedy (one completion per layer, fast)
# True  → sample N completions per layer, report rhyme rate
SAMPLING_MODE = True
SAMPLING_N = 500
SAMPLING_TEMP = 0.8

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
        w
        for w, phones_list in cmu.items()
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
    return last_word(text[: newline_positions[n - 1]])


def extract_rhyme_word(full_text: str, prompt: str) -> str:
    """
    Extract rhyme word from first generated line.
    For this prompt, that is the word before the 3rd newline in full text.
    """
    target_newline_index = prompt.count("\n") + 1
    rhyme_word = word_before_nth_newline(full_text, target_newline_index)
    if rhyme_word:
        return rhyme_word
    if full_text.startswith(prompt):
        return last_word(full_text[len(prompt) :])
    return last_word(full_text)


# ── HF Helpers ──────────────────────────────────────────────────────────────────

def get_model_layers(model):
    """Return the list-like transformer block container for common HF CausalLMs."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported model architecture: could not find transformer layers.")


def to_tokens(tokenizer, text: str, add_special_tokens: bool = True) -> torch.Tensor:
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)
    return enc["input_ids"]


def tokens_to_text(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False)


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float) -> str:
    input_ids = to_tokens(tokenizer, prompt, add_special_tokens=True)
    device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    do_sample = temperature > 0
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def sample_completions(model, tokenizer, prompt: str, n: int, temperature: float) -> list[str]:
    """Draw n sampled completions from the model given a prompt."""
    completions = []
    for _ in range(n):
        completions.append(
            generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=temperature,
            )
        )
    return completions


def rhyme_rate(completions: list[str], prompt: str, rhyme_set: set[str]) -> float:
    """Fraction of completions whose extracted rhyme word is in rhyme_set."""
    hits = sum(extract_rhyme_word(c, prompt) in rhyme_set for c in completions)
    return hits / len(completions) if completions else 0.0


def find_patch_pos(tokenizer, corrupt_tokens_1d: list[int]) -> tuple[int, str]:
    """Find patch position for newline or r1 mode."""
    if PATCH_MODE == "newline":
        newline_ids = tokenizer("\n", add_special_tokens=False)["input_ids"]
        if not newline_ids:
            raise ValueError("Tokenizer produced no token ids for newline.")
        # Use the final token id for newline representation and find its last occurrence.
        newline_id = newline_ids[-1]
        newline_positions = [i for i, t in enumerate(corrupt_tokens_1d) if t == newline_id]
        if newline_positions:
            patch_pos = max(newline_positions)
        else:
            # Fallback to character-to-token mapping.
            last_newline_char = CORRUPT_PROMPT.rfind("\n")
            if last_newline_char == -1:
                raise ValueError("No newline character in corrupt prompt.")
            enc = tokenizer(
                CORRUPT_PROMPT,
                return_offsets_mapping=True,
                add_special_tokens=True,
            )
            offset_mapping = enc.get("offset_mapping") or []
            patch_pos = None
            for i, (start, end) in enumerate(offset_mapping):
                if start <= last_newline_char < end:
                    patch_pos = i
                    break
            if patch_pos is None:
                raise ValueError("Could not find token covering newline in corrupt prompt.")
        patch_label = f"newline (i=0, pos={patch_pos})"
        return patch_pos, patch_label

    if PATCH_MODE == "r1":
        corrupt_r1_ids = tokenizer(
            f" {CORRUPT_RHYME_WORD}",
            add_special_tokens=False,
        )["input_ids"]
        patch_pos = None
        for i in range(len(corrupt_tokens_1d) - len(corrupt_r1_ids), -1, -1):
            if corrupt_tokens_1d[i : i + len(corrupt_r1_ids)] == corrupt_r1_ids:
                patch_pos = i
                break
        if patch_pos is None:
            raise ValueError(f"Could not find '{CORRUPT_RHYME_WORD}' token in corrupt prompt.")
        patch_label = f"r1 token ('{CORRUPT_RHYME_WORD}', pos={patch_pos})"
        return patch_pos, patch_label

    raise ValueError(f"PATCH_MODE must be 'newline' or 'r1', got '{PATCH_MODE}'")


def cache_clean_resid_pre(model, clean_input_ids: torch.Tensor, patch_pos: int) -> list[torch.Tensor]:
    """Cache resid_pre at patch_pos for each layer on the clean prompt."""
    layers = get_model_layers(model)
    device = model.get_input_embeddings().weight.device
    clean_input_ids = clean_input_ids.to(device)
    attention_mask = torch.ones_like(clean_input_ids, device=device)

    cached = [None] * len(layers)
    handles = []

    def make_capture_hook(layer_idx: int):
        def capture_hook(module, args):
            hidden = args[0]
            if hidden.shape[1] > patch_pos:
                cached[layer_idx] = hidden[:, patch_pos, :].detach().clone()
        return capture_hook

    for idx, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(make_capture_hook(idx)))

    try:
        with torch.no_grad():
            model(input_ids=clean_input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    missing = [i for i, v in enumerate(cached) if v is None]
    if missing:
        raise RuntimeError(f"Failed to cache clean resid_pre for layers: {missing}")
    return cached


@contextmanager
def patch_single_layer_resid_pre(model, layer_idx: int, patch_pos: int, clean_vec: torch.Tensor):
    """Temporarily patch resid_pre at one layer and one position."""
    layers = get_model_layers(model)
    layer = layers[layer_idx]

    def patch_hook(module, args):
        hidden = args[0]
        if hidden.shape[1] <= patch_pos:
            return args
        patched = hidden.clone()
        patched[:, patch_pos, :] = clean_vec.to(patched.device, dtype=patched.dtype)
        if len(args) == 1:
            return (patched,)
        return (patched, *args[1:])

    handle = layer.register_forward_pre_hook(patch_hook)
    try:
        yield
    finally:
        handle.remove()


# ── Model Loading ───────────────────────────────────────────────────────────────

def load_model():
    print(f"Loading {MODEL_NAME} via Hugging Face transformers...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    layers = get_model_layers(model)
    d_model = getattr(model.config, "hidden_size", "unknown")
    print(f"Loaded. Layers: {len(layers)} | d_model: {d_model}")
    return model, tokenizer


# ── Main Experiment ─────────────────────────────────────────────────────────────

def run_experiment():
    model, tokenizer = load_model()

    # --- Validate flags ---
    if PATCH_MODE not in ("newline", "r1"):
        raise ValueError(f"PATCH_MODE must be 'newline' or 'r1', got '{PATCH_MODE}'")

    # --- Build rhyme sets ---
    print("\nBuilding rhyme sets from CMU dict...")
    clean_rhymes = build_rhyme_set(CLEAN_RHYME_WORD)
    corrupt_rhymes = build_rhyme_set(CORRUPT_RHYME_WORD)
    overlap = clean_rhymes & corrupt_rhymes
    if overlap:
        print(f"  Removing {len(overlap)} overlapping words from both sets")
        clean_rhymes -= overlap
        corrupt_rhymes -= overlap
    print(f"  '{CLEAN_RHYME_WORD}' rhymes: {len(clean_rhymes)} words, e.g. {list(clean_rhymes)[:6]}")
    print(f"  '{CORRUPT_RHYME_WORD}' rhymes: {len(corrupt_rhymes)} words, e.g. {list(corrupt_rhymes)[:6]}")

    # --- Tokenize ---
    clean_tokens = to_tokens(tokenizer, CLEAN_PROMPT, add_special_tokens=True)
    corrupt_tokens = to_tokens(tokenizer, CORRUPT_PROMPT, add_special_tokens=True)

    if clean_tokens.shape[1] != corrupt_tokens.shape[1]:
        print(f"\nWARNING: token length mismatch ({clean_tokens.shape[1]} vs {corrupt_tokens.shape[1]})")
        print("Prompts should tokenize to the same length for clean positional alignment.")

    tok_list = corrupt_tokens[0].tolist()
    patch_pos, patch_label = find_patch_pos(tokenizer, tok_list)

    print(f"\nPatch mode: {PATCH_MODE} → patching at {patch_label}")
    print(
        f"Sampling mode: {SAMPLING_MODE}"
        + (f" (N={SAMPLING_N}, T={SAMPLING_TEMP})" if SAMPLING_MODE else " (greedy)")
    )
    print("\nCorrupt tokens:")
    for i, tok in enumerate(tok_list):
        marker = f" <-- patch target ({patch_label})" if i == patch_pos else ""
        print(f"  pos {i:2d}: {repr(tokens_to_text(tokenizer, tok))}{marker}")

    # --- Baseline completions ---
    print("\n── Baseline Completions (greedy) ──")
    clean_completion = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=CLEAN_PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0,
    )
    corrupt_completion = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=CORRUPT_PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0,
    )
    print(f"Clean   -> {repr(clean_completion)}")
    print(f"Corrupt -> {repr(corrupt_completion)}")
    clean_end = extract_rhyme_word(clean_completion, CLEAN_PROMPT)
    corrupt_end = extract_rhyme_word(corrupt_completion, CORRUPT_PROMPT)
    print(f"\nClean ends with:   '{clean_end}' — rhymes with '{CLEAN_RHYME_WORD}'?   {clean_end in clean_rhymes}")
    print(f"Corrupt ends with: '{corrupt_end}' — rhymes with '{CORRUPT_RHYME_WORD}'? {corrupt_end in corrupt_rhymes}")

    # --- Sampling baseline (unpatched corrupt run) ---
    if SAMPLING_MODE:
        print(f"\n── Unpatched Corrupt Baseline ({SAMPLING_N} samples, T={SAMPLING_TEMP}) ──")
        baseline_samples = sample_completions(model, tokenizer, CORRUPT_PROMPT, SAMPLING_N, SAMPLING_TEMP)
        baseline_clean_rate = rhyme_rate(baseline_samples, CORRUPT_PROMPT, clean_rhymes)
        baseline_corrupt_rate = rhyme_rate(baseline_samples, CORRUPT_PROMPT, corrupt_rhymes)
        print(f"  Rhymes with '{CLEAN_RHYME_WORD}' (clean): {baseline_clean_rate:.3f}")
        print(f"  Rhymes with '{CORRUPT_RHYME_WORD}' (corrupt): {baseline_corrupt_rate:.3f}")
    else:
        baseline_clean_rate = None
        baseline_corrupt_rate = None

    # --- Cache clean activations ---
    print("\nCaching clean activations...")
    clean_cache = cache_clean_resid_pre(model, clean_tokens, patch_pos)

    # --- Sweep layers ---
    layers = get_model_layers(model)
    n_layers = len(layers)
    print(f"\nPatching at {patch_label} across all {n_layers} layers...")
    print(
        f"Mode: {'sampling (N=' + str(SAMPLING_N) + ', T=' + str(SAMPLING_TEMP) + ')' if SAMPLING_MODE else 'greedy'}\n"
    )

    results = []

    for layer in range(n_layers):
        clean_vec = clean_cache[layer]

        if SAMPLING_MODE:
            with patch_single_layer_resid_pre(model, layer, patch_pos, clean_vec):
                completions = sample_completions(model, tokenizer, CORRUPT_PROMPT, SAMPLING_N, SAMPLING_TEMP)
            clean_rate = rhyme_rate(completions, CORRUPT_PROMPT, clean_rhymes)
            corrupt_rate = rhyme_rate(completions, CORRUPT_PROMPT, corrupt_rhymes)
            result = {
                "layer": layer,
                "completions": completions,
                "clean_rhyme_rate": clean_rate,
                "corrupt_rhyme_rate": corrupt_rate,
                "baseline_clean_rate": baseline_clean_rate,
                "baseline_corrupt_rate": baseline_corrupt_rate,
            }
            delta = clean_rate - baseline_clean_rate
            print(
                f"  Layer {layer:2d}: clean_rhyme_rate={clean_rate:.3f} "
                f"(baseline={baseline_clean_rate:.3f}, delta={delta:+.3f})"
            )
        else:
            with patch_single_layer_resid_pre(model, layer, patch_pos, clean_vec):
                completion = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=CORRUPT_PROMPT,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=0,
                )
            end_word = extract_rhyme_word(completion, CORRUPT_PROMPT)
            rhymes_with_clean = end_word in clean_rhymes
            rhymes_with_corrupt = end_word in corrupt_rhymes
            result = {
                "layer": layer,
                "completion": completion,
                "end_word": end_word,
                "rhymes_with_clean": rhymes_with_clean,
                "rhymes_with_corrupt": rhymes_with_corrupt,
            }
            status = (
                f"✓ rhymes with '{CLEAN_RHYME_WORD}'"
                if rhymes_with_clean
                else f"✗ rhymes with '{CORRUPT_RHYME_WORD}'"
                if rhymes_with_corrupt
                else f"? '{end_word}'"
            )
            print(f"  Layer {layer:2d}: {status}  |  {repr(completion.strip())}")

        results.append(result)

    # --- Summary ---
    print("\n── Summary ──")
    if SAMPLING_MODE:
        best = max(results, key=lambda r: r["clean_rhyme_rate"])
        print(
            f"Best layer: {best['layer']} "
            f"(clean_rhyme_rate={best['clean_rhyme_rate']:.3f}, baseline={baseline_clean_rate:.3f})"
        )
    else:
        n_transferred = sum(r["rhymes_with_clean"] for r in results)
        print(f"Layers where patch transferred clean rhyme plan: {n_transferred} / {n_layers}")
        print(f"Successful layers: {[r['layer'] for r in results if r['rhymes_with_clean']]}")

    # --- Plot ---
    layer_ids = [r["layer"] for r in results]
    fig, ax = plt.subplots(figsize=(14, 4))

    if SAMPLING_MODE:
        clean_rates = [r["clean_rhyme_rate"] for r in results]
        corrupt_rates = [r["corrupt_rhyme_rate"] for r in results]
        ax.bar(
            layer_ids,
            clean_rates,
            color="steelblue",
            edgecolor="white",
            linewidth=0.5,
            label=f"'{CLEAN_RHYME_WORD}'-rhyme rate (patched)",
        )
        ax.plot(
            layer_ids,
            corrupt_rates,
            color="darkorange",
            marker="o",
            markersize=3,
            linewidth=1.0,
            label=f"'{CORRUPT_RHYME_WORD}'-rhyme rate (patched)",
        )
        ax.axhline(
            baseline_clean_rate,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"baseline clean rate ({baseline_clean_rate:.3f})",
        )
        ax.axhline(
            baseline_corrupt_rate,
            color="orange",
            linestyle="--",
            linewidth=1.5,
            label=f"baseline corrupt rate ({baseline_corrupt_rate:.3f})",
        )
        ax.set_ylabel("Rhyme rate")
        ax.legend(loc="upper right")
    else:
        colors = [
            "steelblue"
            if r["rhymes_with_clean"]
            else "salmon"
            if r["rhymes_with_corrupt"]
            else "lightgray"
            for r in results
        ]
        ax.bar(layer_ids, [1] * len(layer_ids), color=colors, edgecolor="white", linewidth=0.5)
        from matplotlib.patches import Patch

        ax.legend(
            handles=[
                Patch(facecolor="steelblue", label=f"Ends with '{CLEAN_RHYME_WORD}'-rhyme (transfer ✓)"),
                Patch(facecolor="salmon", label=f"Ends with '{CORRUPT_RHYME_WORD}'-rhyme (no transfer)"),
                Patch(facecolor="lightgray", label="Neither"),
            ],
            loc="upper right",
        )
        ax.set_yticks([])

    ax.set_xlabel(f"Layer (patch mode: {PATCH_MODE} @ {patch_label})")
    ax.set_xticks(layer_ids)
    ax.set_title(
        f"Does patching [{patch_label}] transfer the rhyme plan? "
        f"({'sampling N=' + str(SAMPLING_N) + ' T=' + str(SAMPLING_TEMP) if SAMPLING_MODE else 'greedy'})\n"
        f"{MODEL_NAME} | clean r1='{CLEAN_RHYME_WORD}' → corrupt run (r1='{CORRUPT_RHYME_WORD}')"
    )
    ax.set_xlim(-0.5, n_layers - 0.5)

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
        "n_layers": n_layers,
        "results": results,
    }
    json_path = os.path.join(run_dir, "generations.json")
    with open(json_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"Generations saved to {json_path}")

    return results


if __name__ == "__main__":
    results = run_experiment()
