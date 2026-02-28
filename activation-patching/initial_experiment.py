import argparse
import json
import os
import re
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import nltk
import torch

# ── Global Defaults ─────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-32B"
SAMPLING_MODE = True
SAMPLING_N = 500
SAMPLING_TEMP = 0.8
MAX_NEW_TOKENS = 20

# Shared prompt pair used by "OG setup" experiments.
OG_CORRUPT_PROMPT = "A rhyming couplet:\nHe felt a sudden urge to sleep,\n"
OG_CLEAN_PROMPT = "A rhyming couplet:\nHe felt a sudden urge to rest,\n"

CORRUPT_RHYME_WORD = "sleep"
CLEAN_RHYME_WORD = "rest"


def patch_selector_newline() -> dict[str, str]:
    return {"kind": "newline"}


def patch_selector_token(token_text: str) -> dict[str, str]:
    return {"kind": "token_text", "token_text": token_text}


EXPERIMENT_SPECS: list[dict[str, Any]] = [
    # Exp 1: extended prompt; separate sweeps for newline / But / then / he.
    {
        "id": "exp1_newline",
        "run_name": "qwen2.5_32b_exp1_extended_corrupt_to_clean_newline",
        "corrupt_prompt": "A rhyming couplet:\nHe felt a sudden urge to sleep,\nBut then he",
        "clean_prompt": "A rhyming couplet:\nHe felt a sudden urge to rest,\nBut then he",
        "patch_source_mode": "corrupt_to_clean",
        "target_selector": patch_selector_newline(),
        "source_selector": patch_selector_newline(),
    },
    {
        "id": "exp1_but",
        "run_name": "qwen2.5_32b_exp1_extended_corrupt_to_clean_but",
        "corrupt_prompt": "A rhyming couplet:\nHe felt a sudden urge to sleep,\nBut then he",
        "clean_prompt": "A rhyming couplet:\nHe felt a sudden urge to rest,\nBut then he",
        "patch_source_mode": "corrupt_to_clean",
        "target_selector": patch_selector_token(" But"),
        "source_selector": patch_selector_token(" But"),
    },
    {
        "id": "exp1_then",
        "run_name": "qwen2.5_32b_exp1_extended_corrupt_to_clean_then",
        "corrupt_prompt": "A rhyming couplet:\nHe felt a sudden urge to sleep,\nBut then he",
        "clean_prompt": "A rhyming couplet:\nHe felt a sudden urge to rest,\nBut then he",
        "patch_source_mode": "corrupt_to_clean",
        "target_selector": patch_selector_token(" then"),
        "source_selector": patch_selector_token(" then"),
    },
    {
        "id": "exp1_he",
        "run_name": "qwen2.5_32b_exp1_extended_corrupt_to_clean_he",
        "corrupt_prompt": "A rhyming couplet:\nHe felt a sudden urge to sleep,\nBut then he",
        "clean_prompt": "A rhyming couplet:\nHe felt a sudden urge to rest,\nBut then he",
        "patch_source_mode": "corrupt_to_clean",
        "target_selector": patch_selector_token(" he"),
        "source_selector": patch_selector_token(" he"),
    },
    # Exp 2: asymmetric context, patch newline.
    {
        "id": "exp2_newline_asymmetric_context",
        "run_name": "qwen2.5_32b_exp2_newline_asymmetric_corrupt_to_clean",
        "corrupt_prompt": "A rhyming couplet:\nHe felt a sudden urge to sleep,\nBut then he",
        "clean_prompt": "He felt a sudden urge to rest,\nBut then he",
        "patch_source_mode": "corrupt_to_clean",
        "target_selector": patch_selector_newline(),
        "source_selector": patch_selector_newline(),
    },
    # Exp 3: OG setup, zero vector at clean newline.
    {
        "id": "exp3_og_clean_newline_zero",
        "run_name": "qwen2.5_32b_exp3_og_clean_newline_zero",
        "corrupt_prompt": OG_CORRUPT_PROMPT,
        "clean_prompt": OG_CLEAN_PROMPT,
        "patch_source_mode": "zero_vector",
        "target_selector": patch_selector_newline(),
    },
    # Exp 4: OG setup, donor newline.
    {
        "id": "exp4_og_clean_newline_donor",
        "run_name": "qwen2.5_32b_exp4_og_clean_newline_donor",
        "corrupt_prompt": OG_CORRUPT_PROMPT,
        "clean_prompt": OG_CLEAN_PROMPT,
        "patch_source_mode": "donor_prompt",
        "target_selector": patch_selector_newline(),
        "donor_prompt": "Hello, how are you doing today?\n",
        "donor_selector": patch_selector_newline(),
    },
]


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
        if w != word.lower() and any(get_rhyme_tail(p) == target_tail for p in phones_list)
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
    """Extract rhyme word from first generated line."""
    target_newline_index = prompt.count("\n") + 1
    rhyme_word = word_before_nth_newline(full_text, target_newline_index)
    if rhyme_word:
        return rhyme_word
    if full_text.startswith(prompt):
        return last_word(full_text[len(prompt) :])
    return last_word(full_text)


def sample_completions(
    model,
    prompt: str,
    n: int,
    temperature: float,
    max_new_tokens: int,
) -> list[str]:
    """Draw n sampled completions from the model given a prompt."""
    completions: list[str] = []
    for _ in range(n):
        completion = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        completions.append(completion)
    return completions


def rhyme_rate(completions: list[str], prompt: str, rhyme_set: set[str]) -> float:
    """Fraction of completions whose extracted rhyme word is in rhyme_set."""
    hits = sum(extract_rhyme_word(c, prompt) in rhyme_set for c in completions)
    return hits / len(completions) if completions else 0.0


# ── Position Resolution ─────────────────────────────────────────────────────────

def find_last_subsequence(sequence: list[int], subseq: list[int]) -> int | None:
    if not subseq or len(subseq) > len(sequence):
        return None
    for i in range(len(sequence) - len(subseq), -1, -1):
        if sequence[i : i + len(subseq)] == subseq:
            return i
    return None


def resolve_newline_pos(model, prompt: str, tok_list: list[int]) -> int:
    newline_ids = model.to_tokens("\n", prepend_bos=False)[0].tolist()
    if len(newline_ids) == 1:
        newline_id = newline_ids[0]
        newline_positions = [i for i, tok in enumerate(tok_list) if tok == newline_id]
        if newline_positions:
            return max(newline_positions)

    last_newline_char = prompt.rfind("\n")
    if last_newline_char == -1:
        raise ValueError("No newline character in prompt.")

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("No tokenizer available for fallback offset mapping.")

    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    offset_mapping = enc.get("offset_mapping") or []
    for i, (start, end) in enumerate(offset_mapping):
        if start <= last_newline_char < end:
            return i
    raise ValueError("Could not find token covering final newline in prompt.")


def resolve_patch_position(
    model,
    prompt: str,
    tok_list: list[int],
    selector: dict[str, str],
) -> tuple[int, str]:
    kind = selector.get("kind")
    if kind == "newline":
        pos = resolve_newline_pos(model, prompt, tok_list)
        return pos, f"newline (pos={pos})"

    if kind == "token_text":
        token_text = selector.get("token_text", "")
        token_ids = model.to_tokens(token_text, prepend_bos=False)[0].tolist()
        if not token_ids:
            raise ValueError(f"Token selector text produced no tokens: {token_text!r}")
        pos = find_last_subsequence(tok_list, token_ids)
        if pos is None:
            raise ValueError(f"Could not find token text {token_text!r} in prompt.")
        return pos, f"token {token_text!r} (pos={pos})"

    raise ValueError(f"Unknown selector kind: {kind!r}")


# ── Model Loading ───────────────────────────────────────────────────────────────

def load_model():
    from transformer_lens import HookedTransformer

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


def sanitize_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_")


# ── Main Experiment Runner ──────────────────────────────────────────────────────

def run_single_experiment(
    model,
    spec: dict[str, Any],
    sampling_mode: bool,
    sampling_n: int,
    sampling_temp: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    run_name = spec["run_name"]
    corrupt_prompt = spec["corrupt_prompt"]
    clean_prompt = spec["clean_prompt"]
    patch_source_mode = spec["patch_source_mode"]
    target_selector = spec["target_selector"]
    source_selector = spec.get("source_selector", target_selector)
    donor_selector = spec.get("donor_selector", patch_selector_newline())

    print("\n" + "=" * 100)
    print(f"Running {spec['id']} -> {run_name}")
    print(f"Patch source mode: {patch_source_mode}")
    print("=" * 100)

    print("\nBuilding rhyme sets from CMU dict...")
    clean_rhymes = build_rhyme_set(CLEAN_RHYME_WORD)
    corrupt_rhymes = build_rhyme_set(CORRUPT_RHYME_WORD)
    overlap = clean_rhymes & corrupt_rhymes
    if overlap:
        print(f"  Removing {len(overlap)} overlapping words from both sets")
        clean_rhymes -= overlap
        corrupt_rhymes -= overlap
    print(f"  '{CLEAN_RHYME_WORD}' rhymes: {len(clean_rhymes)} words")
    print(f"  '{CORRUPT_RHYME_WORD}' rhymes: {len(corrupt_rhymes)} words")

    clean_tokens = model.to_tokens(clean_prompt)
    corrupt_tokens = model.to_tokens(corrupt_prompt)
    clean_tok_list = clean_tokens[0].tolist()
    corrupt_tok_list = corrupt_tokens[0].tolist()

    target_pos, target_label = resolve_patch_position(model, clean_prompt, clean_tok_list, target_selector)
    print(f"\nTarget position on clean run: {target_label}")

    if clean_tokens.shape[1] != corrupt_tokens.shape[1]:
        print(
            f"WARNING: token length mismatch "
            f"(clean={clean_tokens.shape[1]}, corrupt={corrupt_tokens.shape[1]})."
        )

    source_cache = None
    source_pos = None
    source_label = None
    source_prompt = None

    if patch_source_mode == "corrupt_to_clean":
        source_prompt = corrupt_prompt
        source_pos, source_label = resolve_patch_position(
            model, source_prompt, corrupt_tok_list, source_selector
        )
        print(f"Source position on corrupt run: {source_label}")
        print("Caching corrupt activations (source run)...")
        _, source_cache = model.run_with_cache(source_prompt)
    elif patch_source_mode == "zero_vector":
        print("Using zero-vector source at target position.")
    elif patch_source_mode == "donor_prompt":
        source_prompt = spec["donor_prompt"]
        donor_tokens = model.to_tokens(source_prompt)
        donor_tok_list = donor_tokens[0].tolist()
        source_pos, source_label = resolve_patch_position(
            model, source_prompt, donor_tok_list, donor_selector
        )
        print(f"Source position on donor prompt: {source_label}")
        print("Caching donor activations (source run)...")
        _, source_cache = model.run_with_cache(source_prompt)
    else:
        raise ValueError(f"Unknown patch_source_mode: {patch_source_mode!r}")

    print("\n── Baseline Completions (greedy) ──")
    clean_completion = model.generate(clean_prompt, max_new_tokens=max_new_tokens, temperature=0)
    corrupt_completion = model.generate(corrupt_prompt, max_new_tokens=max_new_tokens, temperature=0)
    print(f"Clean   -> {repr(clean_completion)}")
    print(f"Corrupt -> {repr(corrupt_completion)}")
    clean_end = extract_rhyme_word(clean_completion, clean_prompt)
    corrupt_end = extract_rhyme_word(corrupt_completion, corrupt_prompt)
    print(f"Clean ends with '{clean_end}', in clean rhyme set? {clean_end in clean_rhymes}")
    print(f"Corrupt ends with '{corrupt_end}', in corrupt rhyme set? {corrupt_end in corrupt_rhymes}")

    if sampling_mode:
        print(f"\n── Unpatched Clean Baseline ({sampling_n} samples, T={sampling_temp}) ──")
        baseline_samples = sample_completions(
            model=model,
            prompt=clean_prompt,
            n=sampling_n,
            temperature=sampling_temp,
            max_new_tokens=max_new_tokens,
        )
        baseline_clean_rate = rhyme_rate(baseline_samples, clean_prompt, clean_rhymes)
        baseline_corrupt_rate = rhyme_rate(baseline_samples, clean_prompt, corrupt_rhymes)
        print(f"  clean_rhyme_rate={baseline_clean_rate:.3f}")
        print(f"  corrupt_rhyme_rate={baseline_corrupt_rate:.3f}")
    else:
        baseline_clean_rate = None
        baseline_corrupt_rate = None

    print(
        f"\nPatching clean target at {target_label} across all {model.cfg.n_layers} layers "
        f"(source mode: {patch_source_mode})..."
    )

    results: list[dict[str, Any]] = []
    for layer in range(model.cfg.n_layers):
        if patch_source_mode in ("corrupt_to_clean", "donor_prompt"):
            source_vec = source_cache[f"blocks.{layer}.hook_resid_pre"][:, source_pos, :].clone()
        else:
            source_vec = None

        def patch_hook(value, hook, vec=source_vec, target_pos_=target_pos):
            if value.shape[1] <= target_pos_:
                return value
            out = value.clone()
            if vec is None:
                out[:, target_pos_, :] = 0.0
            else:
                out[:, target_pos_, :] = vec
            return out

        hook = (f"blocks.{layer}.hook_resid_pre", patch_hook)

        if sampling_mode:
            with model.hooks(fwd_hooks=[hook]):
                completions = sample_completions(
                    model=model,
                    prompt=clean_prompt,
                    n=sampling_n,
                    temperature=sampling_temp,
                    max_new_tokens=max_new_tokens,
                )
            clean_rate = rhyme_rate(completions, clean_prompt, clean_rhymes)
            corrupt_rate = rhyme_rate(completions, clean_prompt, corrupt_rhymes)
            result = {
                "layer": layer,
                "completions": completions,
                "clean_rhyme_rate": clean_rate,
                "corrupt_rhyme_rate": corrupt_rate,
                "baseline_clean_rate": baseline_clean_rate,
                "baseline_corrupt_rate": baseline_corrupt_rate,
            }
            delta = corrupt_rate - baseline_corrupt_rate
            print(
                f"  Layer {layer:2d}: corrupt_rhyme_rate={corrupt_rate:.3f} "
                f"(baseline={baseline_corrupt_rate:.3f}, delta={delta:+.3f})"
            )
        else:
            with model.hooks(fwd_hooks=[hook]):
                completion = model.generate(clean_prompt, max_new_tokens=max_new_tokens, temperature=0)
            end_word = extract_rhyme_word(completion, clean_prompt)
            rhymes_with_clean = end_word in clean_rhymes
            rhymes_with_corrupt = end_word in corrupt_rhymes
            result = {
                "layer": layer,
                "completion": completion,
                "end_word": end_word,
                "rhymes_with_clean": rhymes_with_clean,
                "rhymes_with_corrupt": rhymes_with_corrupt,
            }
            print(
                f"  Layer {layer:2d}: end_word='{end_word}' "
                f"(clean={rhymes_with_clean}, corrupt={rhymes_with_corrupt})"
            )

        results.append(result)

    print("\n── Summary ──")
    if sampling_mode:
        best = max(results, key=lambda r: r["corrupt_rhyme_rate"])
        print(
            f"Best layer: {best['layer']} "
            f"(corrupt_rhyme_rate={best['corrupt_rhyme_rate']:.3f}, "
            f"baseline={baseline_corrupt_rate:.3f})"
        )
    else:
        n_transferred = sum(r["rhymes_with_corrupt"] for r in results)
        print(f"Layers where patch transferred corrupt rhyme plan: {n_transferred} / {model.cfg.n_layers}")

    layers = [r["layer"] for r in results]
    fig, ax = plt.subplots(figsize=(14, 4))
    if sampling_mode:
        clean_rates = [r["clean_rhyme_rate"] for r in results]
        corrupt_rates = [r["corrupt_rhyme_rate"] for r in results]
        ax.bar(
            layers,
            corrupt_rates,
            color="darkorange",
            edgecolor="white",
            linewidth=0.5,
            label=f"'{CORRUPT_RHYME_WORD}' rhyme rate (patched)",
        )
        ax.plot(
            layers,
            clean_rates,
            color="steelblue",
            marker="o",
            markersize=3,
            linewidth=1.0,
            label=f"'{CLEAN_RHYME_WORD}' rhyme rate (patched)",
        )
        ax.axhline(
            baseline_corrupt_rate,
            color="orange",
            linestyle="--",
            linewidth=1.5,
            label=f"baseline corrupt rate ({baseline_corrupt_rate:.3f})",
        )
        ax.axhline(
            baseline_clean_rate,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"baseline clean rate ({baseline_clean_rate:.3f})",
        )
        ax.set_ylabel("Rhyme rate")
        ax.legend(loc="upper right")
    else:
        colors = [
            "darkorange" if r["rhymes_with_corrupt"] else "steelblue" if r["rhymes_with_clean"] else "lightgray"
            for r in results
        ]
        ax.bar(layers, [1] * len(layers), color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks([])

    ax.set_xlabel(f"Layer (target: {target_label})")
    ax.set_xticks(layers)
    ax.set_title(
        f"Patch mode: {patch_source_mode} | {MODEL_NAME}\n"
        f"corrupt='{CORRUPT_RHYME_WORD}' -> clean='{CLEAN_RHYME_WORD}'"
    )
    ax.set_xlim(-0.5, model.cfg.n_layers - 0.5)
    plt.tight_layout()

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    run_dir = os.path.join(results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    selector_slug = sanitize_name(target_label)
    mode_slug = "sampling" if sampling_mode else "greedy"
    image_path = os.path.join(run_dir, f"patching_results_{selector_slug}_{mode_slug}.png")
    plt.savefig(image_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {image_path}")
    plt.close(fig)

    export = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_id": spec["id"],
        "run_name": run_name,
        "model_name": MODEL_NAME,
        "patch_source_mode": patch_source_mode,
        "target_selector": target_selector,
        "target_patch_label": target_label,
        "target_patch_pos": int(target_pos),
        "source_selector": source_selector if patch_source_mode == "corrupt_to_clean" else None,
        "source_patch_label": source_label,
        "source_patch_pos": None if source_pos is None else int(source_pos),
        "donor_prompt": spec.get("donor_prompt"),
        "sampling_mode": sampling_mode,
        "sampling_n": sampling_n if sampling_mode else None,
        "sampling_temp": sampling_temp if sampling_mode else None,
        "max_new_tokens": max_new_tokens,
        "clean_prompt": clean_prompt,
        "corrupt_prompt": corrupt_prompt,
        "clean_rhyme_word": CLEAN_RHYME_WORD,
        "corrupt_rhyme_word": CORRUPT_RHYME_WORD,
        "baseline": {
            "clean_completion": clean_completion,
            "corrupt_completion": corrupt_completion,
            "unpatched_clean_clean_rhyme_rate": baseline_clean_rate,
            "unpatched_clean_corrupt_rhyme_rate": baseline_corrupt_rate,
        },
        "n_layers": model.cfg.n_layers,
        "results": results,
    }
    json_path = os.path.join(run_dir, "generations.json")
    with open(json_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"Generations saved to {json_path}")

    return {
        "experiment_id": spec["id"],
        "run_name": run_name,
        "results_dir": run_dir,
        "image_path": image_path,
        "json_path": json_path,
        "target_label": target_label,
        "source_label": source_label,
    }


def get_spec_by_id(experiment_id: str) -> dict[str, Any]:
    for spec in EXPERIMENT_SPECS:
        if spec["id"] == experiment_id:
            return spec
    raise ValueError(f"Unknown experiment id: {experiment_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Activation patching experiments (corrupt -> clean).")
    parser.add_argument(
        "--experiment-id",
        type=str,
        default="all",
        help="Experiment id to run, or 'all'.",
    )
    parser.add_argument("--sampling-n", type=int, default=SAMPLING_N)
    parser.add_argument("--sampling-temp", type=float, default=SAMPLING_TEMP)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding instead of sampling mode.",
    )
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List experiment ids and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate selected specs and print config without loading model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sampling_mode = not args.greedy

    if args.list_experiments:
        print("Available experiment ids:")
        for spec in EXPERIMENT_SPECS:
            print(f"  - {spec['id']}: {spec['run_name']}")
        return

    if args.experiment_id == "all":
        specs = EXPERIMENT_SPECS
    else:
        specs = [get_spec_by_id(args.experiment_id)]

    print(f"Selected experiments: {[s['id'] for s in specs]}")
    if args.dry_run:
        print("Dry run selected. Spec summary:")
        for spec in specs:
            print(
                f"  {spec['id']}: mode={spec['patch_source_mode']}, "
                f"target={spec['target_selector']}, run_name={spec['run_name']}"
            )
        return

    model = load_model()
    run_outputs: list[dict[str, Any]] = []
    for spec in specs:
        output = run_single_experiment(
            model=model,
            spec=spec,
            sampling_mode=sampling_mode,
            sampling_n=args.sampling_n,
            sampling_temp=args.sampling_temp,
            max_new_tokens=args.max_new_tokens,
        )
        run_outputs.append(output)

    print("\nAll requested experiments finished.")
    for out in run_outputs:
        print(f"- {out['experiment_id']}: {out['results_dir']}")


if __name__ == "__main__":
    main()