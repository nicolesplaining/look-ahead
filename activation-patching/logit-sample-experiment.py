import json
import os
import torch
import pronouncing
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ─────────────────────────────────────────────────────────────────────

RUN_NAME = "qwen3-32b-logit-sample"

MODEL_NAME = "Qwen/Qwen3-32B"

# Patch direction: corrupt → clean (inject "sleep" activations into "rest" run)
CLEAN_PROMPT   = "A rhyming couplet:\nHe felt a sudden urge to rest,\n"
CORRUPT_PROMPT = "A rhyming couplet:\nHe felt a sudden urge to sleep,\n"

CLEAN_RHYME_WORD   = "rest"
CORRUPT_RHYME_WORD = "sleep"

SAMPLING_N    = 50
SAMPLING_TEMP = 0.7
MAX_NEW_TOKENS = 20

# ── Rhyme Token Lookup ──────────────────────────────────────────────────────────

def get_rhyme_token_ids(rhyme_word: str, tokenizer) -> list[int]:
    """Return vocab IDs for all single-token words that rhyme with rhyme_word."""
    token_ids = set()
    for word in pronouncing.rhymes(rhyme_word):
        for variant in [f" {word}", word, f" {word.capitalize()}", word.capitalize()]:
            ids = tokenizer(variant, add_special_tokens=False).input_ids
            if len(ids) == 1:
                token_ids.add(ids[0])
                break
    return list(token_ids)

def prob_mass(logits: torch.Tensor, token_ids: list[int]) -> float:
    """Total softmax probability mass on a set of token IDs."""
    if not token_ids:
        return 0.0
    return torch.log_softmax(logits, dim=-1)[token_ids].exp().sum().item()

# ── Model Loading ───────────────────────────────────────────────────────────────

def load_model():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Loaded. Layers: {model.config.num_hidden_layers} | d_model: {model.config.hidden_size}")
    return model, tokenizer

# ── Forward Pass Helpers ────────────────────────────────────────────────────────

def get_input_device(model) -> torch.device:
    return model.model.embed_tokens.weight.device

def cache_hidden_states(model, tokenizer, prompt: str) -> tuple:
    """
    hidden_states[L] = resid_pre for layer L (maps to TL's blocks.{L}.hook_resid_pre)
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

# ── Rhyme Step Finding ──────────────────────────────────────────────────────────

def find_rhyme_step(generated_ids: torch.Tensor, tokenizer) -> int | None:
    """
    Find generation step index t such that scores[t] = logits at the rhyme word position.
    The rhyme word is the last alphabetic token before the first newline in generated_ids.
    Returns None if no valid position found.
    """
    id_list = generated_ids.tolist()

    # Find first newline step
    newline_step = None
    for i, tok_id in enumerate(id_list):
        if "\n" in tokenizer.decode([tok_id]):
            newline_step = i
            break

    search_end = newline_step if newline_step is not None else len(id_list)

    # Walk backwards to find last alphabetic token before newline
    for i in range(search_end - 1, -1, -1):
        if any(c.isalpha() for c in tokenizer.decode([id_list[i]])):
            return i

    return None

# ── Generation with Logit Capture ──────────────────────────────────────────────

def sample_rhyme_probs(
    model, tokenizer, prompt: str, temperature: float,
    clean_tok_ids: list[int], corrupt_tok_ids: list[int],
) -> tuple[float, float] | None:
    """
    Generate one completion. Compute and return (clean_prob_mass, corrupt_prob_mass)
    at the rhyme word position. Logits are discarded after prob_mass is computed.
    Returns None if no rhyme position found.
    """
    device = get_input_device(model)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = enc.input_ids.shape[1]

    with torch.no_grad():
        output = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            output_scores=True,
            return_dict_in_generate=True,
        )

    generated_ids = output.sequences[0][prompt_len:]
    rhyme_step = find_rhyme_step(generated_ids, tokenizer)
    if rhyme_step is None:
        return None

    logits = output.scores[rhyme_step][0].float().cpu()
    return prob_mass(logits, clean_tok_ids), prob_mass(logits, corrupt_tok_ids)

def collect_rhyme_probs(
    model, tokenizer, prompt: str, n: int, temperature: float,
    clean_tok_ids: list[int], corrupt_tok_ids: list[int],
) -> tuple[list[float], list[float]]:
    """Draw n samples; return (clean_masses, corrupt_masses) lists."""
    clean_masses, corrupt_masses = [], []
    for _ in range(n):
        result = sample_rhyme_probs(model, tokenizer, prompt, temperature, clean_tok_ids, corrupt_tok_ids)
        if result is not None:
            clean_masses.append(result[0])
            corrupt_masses.append(result[1])
    return clean_masses, corrupt_masses

# ── Main Experiment ─────────────────────────────────────────────────────────────

def run_experiment():
    model, tokenizer = load_model()
    n_layers = model.config.num_hidden_layers

    # --- Tokenize & find patch position ---
    clean_ids   = tokenizer(CLEAN_PROMPT,   return_tensors="pt").input_ids
    corrupt_ids = tokenizer(CORRUPT_PROMPT, return_tensors="pt").input_ids
    if clean_ids.shape[1] != corrupt_ids.shape[1]:
        print(f"\nWARNING: token length mismatch ({clean_ids.shape[1]} vs {corrupt_ids.shape[1]})")
        print("Prompts should tokenize to the same length for correct positional alignment.")

    last_newline_char = CLEAN_PROMPT.rfind("\n")
    enc = tokenizer(CLEAN_PROMPT, return_offsets_mapping=True, add_special_tokens=True)
    patch_pos = next(
        (i for i, (s, e) in enumerate(enc["offset_mapping"]) if s <= last_newline_char < e),
        None,
    )
    if patch_pos is None:
        raise ValueError("Could not find token covering last newline in clean prompt.")
    patch_label = f"newline (pos={patch_pos})"

    # --- Build rhyme token sets ---
    print("\nBuilding rhyme token sets...")
    clean_tok_ids   = get_rhyme_token_ids(CLEAN_RHYME_WORD,   tokenizer)
    corrupt_tok_ids = get_rhyme_token_ids(CORRUPT_RHYME_WORD, tokenizer)
    print(f"  '{CLEAN_RHYME_WORD}' rhyme tokens:   {len(clean_tok_ids)}")
    print(f"  '{CORRUPT_RHYME_WORD}' rhyme tokens: {len(corrupt_tok_ids)}")

    print(f"\nPatch direction: corrupt → clean")
    print(f"  Injecting '{CORRUPT_RHYME_WORD}' activations into '{CLEAN_RHYME_WORD}' run at {patch_label}")
    print(f"  N={SAMPLING_N}, T={SAMPLING_TEMP}")

    print(f"\nClean prompt tokens:")
    for i, tok_id in enumerate(clean_ids[0].tolist()):
        marker = " <-- patch target" if i == patch_pos else ""
        print(f"  pos {i:2d}: {repr(tokenizer.decode([tok_id]))}{marker}")

    # --- Baseline: unpatched clean run ---
    print(f"\n── Unpatched Clean Baseline ({SAMPLING_N} samples, T={SAMPLING_TEMP}) ──")
    base_clean_masses, base_corrupt_masses = collect_rhyme_probs(
        model, tokenizer, CLEAN_PROMPT, SAMPLING_N, SAMPLING_TEMP,
        clean_tok_ids, corrupt_tok_ids,
    )
    baseline_clean_mass   = sum(base_clean_masses)   / len(base_clean_masses)   if base_clean_masses   else 0.0
    baseline_corrupt_mass = sum(base_corrupt_masses) / len(base_corrupt_masses) if base_corrupt_masses else 0.0
    print(f"  Mean P('{CLEAN_RHYME_WORD}' rhymes)   (expected high): {baseline_clean_mass:.4f}")
    print(f"  Mean P('{CORRUPT_RHYME_WORD}' rhymes) (expected low):  {baseline_corrupt_mass:.4f}")

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
            clean_masses, corrupt_masses = collect_rhyme_probs(
                model, tokenizer, CLEAN_PROMPT, SAMPLING_N, SAMPLING_TEMP,
                clean_tok_ids, corrupt_tok_ids,
            )
        finally:
            handle.remove()

        mean_clean   = sum(clean_masses)   / len(clean_masses)   if clean_masses   else 0.0
        mean_corrupt = sum(corrupt_masses) / len(corrupt_masses) if corrupt_masses else 0.0
        delta   = mean_corrupt - baseline_corrupt_mass
        n_valid = len(clean_masses)

        print(f"  Layer {layer:2d}: Δcorrupt={delta:+.4f}  corrupt={mean_corrupt:.4f}  clean={mean_clean:.4f}  (n_valid={n_valid})")
        results.append({
            "layer":                      layer,
            "mean_clean_prob_mass":       mean_clean,
            "mean_corrupt_prob_mass":     mean_corrupt,
            "baseline_clean_prob_mass":   baseline_clean_mass,
            "baseline_corrupt_prob_mass": baseline_corrupt_mass,
            "delta_corrupt_mass":         delta,
            "n_valid":                    n_valid,
        })

    # --- Summary ---
    best = max(results, key=lambda r: r["mean_corrupt_prob_mass"])
    print(f"\nBest layer: {best['layer']} (corrupt_mass={best['mean_corrupt_prob_mass']:.4f}, baseline={baseline_corrupt_mass:.4f})")

    # --- Plot ---
    layers         = [r["layer"]                 for r in results]
    corrupt_masses = [r["mean_corrupt_prob_mass"] for r in results]
    clean_masses_p = [r["mean_clean_prob_mass"]   for r in results]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(layers, corrupt_masses, color="steelblue", edgecolor="white", linewidth=0.5,
           label=f"mean P('{CORRUPT_RHYME_WORD}'-rhyme) (patched)")
    ax.plot(layers, clean_masses_p, color="darkorange", marker="o", markersize=3, linewidth=1.0,
            label=f"mean P('{CLEAN_RHYME_WORD}'-rhyme) (patched)")
    ax.axhline(baseline_corrupt_mass, color="red", linestyle="--", linewidth=1.5,
               label=f"baseline corrupt mass ({baseline_corrupt_mass:.4f})")
    ax.axhline(baseline_clean_mass, color="orange", linestyle="--", linewidth=1.5,
               label=f"baseline clean mass ({baseline_clean_mass:.4f})")
    ax.set_xlabel(f"Layer (patch: {patch_label})")
    ax.set_ylabel("Mean prob mass on rhyme tokens")
    ax.set_xticks(layers)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_title(
        f"Logit-sample patching [{patch_label}]: corrupt ('{CORRUPT_RHYME_WORD}') → clean ('{CLEAN_RHYME_WORD}') run\n"
        f"{MODEL_NAME} | N={SAMPLING_N}, T={SAMPLING_TEMP}"
    )
    ax.legend(loc="upper right")
    plt.tight_layout()

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    run_dir     = os.path.join(results_dir, RUN_NAME)
    os.makedirs(run_dir, exist_ok=True)

    image_path = os.path.join(run_dir, "logit_sample_results.png")
    plt.savefig(image_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {image_path}")

    # --- Save JSON ---
    json_path = os.path.join(run_dir, "logit_sample_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "run_name":               RUN_NAME,
            "model_name":             MODEL_NAME,
            "patch_direction":        "corrupt→clean",
            "patch_label":            patch_label,
            "patch_pos":              int(patch_pos),
            "clean_prompt":           CLEAN_PROMPT,
            "corrupt_prompt":         CORRUPT_PROMPT,
            "clean_rhyme_word":       CLEAN_RHYME_WORD,
            "corrupt_rhyme_word":     CORRUPT_RHYME_WORD,
            "n_clean_rhyme_tokens":   len(clean_tok_ids),
            "n_corrupt_rhyme_tokens": len(corrupt_tok_ids),
            "sampling_n":             SAMPLING_N,
            "sampling_temp":          SAMPLING_TEMP,
            "max_new_tokens":         MAX_NEW_TOKENS,
            "baseline": {
                "mean_clean_prob_mass":   baseline_clean_mass,
                "mean_corrupt_prob_mass": baseline_corrupt_mass,
                "n_valid":                len(base_clean_masses),
            },
            "n_layers": n_layers,
            "results":  results,
        }, f, indent=2)
    print(f"Results saved to {json_path}")

    return results


if __name__ == "__main__":
    results = run_experiment()
