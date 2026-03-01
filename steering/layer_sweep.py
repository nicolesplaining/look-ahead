"""
Layer sweep: apply steering vector at each layer independently (fixed alpha),
measure "-eep" and "-est" rhyme rates as a function of layer.

Usage:
    python layer_sweep.py [--alpha 1.0] [--n_samples 200] [--model_name ...]
"""
import json
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import (
    MODEL_NAME, CLEAN_PROMPT, CORRUPT_PROMPT, RESULTS_DIR, FIGURES_DIR,
    load_model_and_tokenizer, load_cmu_dict, build_rhyme_checkers,
    extract_steering_vector, run_steered_generation, run_baseline_generation,
    evaluate_completions,
)

# ── Parameters ─────────────────────────────────────────────────────────────────

NEUTRAL_PROMPT = "He walked out into the open air,\n"

ALPHA          = 1.0
N_SAMPLES      = 200
TEMPERATURE    = 0.8
MAX_NEW_TOKENS = 20
TOP_P          = 0.95

# ── Sweep ───────────────────────────────────────────────────────────────────────

def run(model, tokenizer, steering_vectors, rhymes_with_eep, rhymes_with_est, num_layers):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Running baseline...")
    baseline = run_baseline_generation(
        model, tokenizer, NEUTRAL_PROMPT, N_SAMPLES, TEMPERATURE, MAX_NEW_TOKENS, TOP_P
    )
    b_clean, b_corrupt, b_neither, _, _ = evaluate_completions(baseline, rhymes_with_eep, rhymes_with_est)
    print(f"Baseline  |  clean={b_clean:.4f}  corrupt={b_corrupt:.4f}  neither={b_neither:.4f}")

    results = {
        "model_name":            MODEL_NAME,
        "neutral_prompt":        NEUTRAL_PROMPT,
        "clean_prompt":          CLEAN_PROMPT,
        "corrupt_prompt":        CORRUPT_PROMPT,
        "alpha":                 ALPHA,
        "n_samples":             N_SAMPLES,
        "temperature":           TEMPERATURE,
        "max_new_tokens":        MAX_NEW_TOKENS,
        "top_p":                 TOP_P,
        "baseline_clean_rate":   b_clean,
        "baseline_corrupt_rate": b_corrupt,
        "baseline_neither_rate": b_neither,
        "baseline_completions":  baseline,
        "layer_sweep":           [],
    }

    for layer in tqdm(range(num_layers), desc="Layer sweep"):
        completions = run_steered_generation(
            model, tokenizer, NEUTRAL_PROMPT, layer, ALPHA, steering_vectors,
            N_SAMPLES, TEMPERATURE, MAX_NEW_TOKENS, TOP_P
        )
        clean_rate, corrupt_rate, neither_rate, clean_words, corrupt_words = \
            evaluate_completions(completions, rhymes_with_eep, rhymes_with_est)

        results["layer_sweep"].append({
            "layer":         layer,
            "clean_rate":    clean_rate,
            "corrupt_rate":  corrupt_rate,
            "neither_rate":  neither_rate,
            "clean_words":   clean_words,
            "corrupt_words": corrupt_words,
        })
        print(f"Layer {layer:2d}  |  clean={clean_rate:.4f}  corrupt={corrupt_rate:.4f}  neither={neither_rate:.4f}")

        with open(os.path.join(RESULTS_DIR, "layer_sweep_partial.json"), "w") as f:
            json.dump(results, f, indent=2)

    return results

# ── Plot ────────────────────────────────────────────────────────────────────────

def plot(results):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    layers        = [r["layer"]        for r in results["layer_sweep"]]
    clean_rates   = [r["clean_rate"]   for r in results["layer_sweep"]]
    corrupt_rates = [r["corrupt_rate"] for r in results["layer_sweep"]]
    b_clean   = results["baseline_clean_rate"]
    b_corrupt = results["baseline_corrupt_rate"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, clean_rates,   color="steelblue", linewidth=1.5, label='"-eep" rate (target)')
    ax.plot(layers, corrupt_rates, color="orange",    linewidth=1.5, label='"-est" rate (control)')
    ax.axhline(b_clean,   color="steelblue", linestyle="--", linewidth=1.0, alpha=0.6,
               label=f'Baseline "-eep" = {b_clean:.3f}')
    ax.axhline(b_corrupt, color="orange",    linestyle="--", linewidth=1.0, alpha=0.6,
               label=f'Baseline "-est" = {b_corrupt:.3f}')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Rhyme Rate")
    ax.set_title(
        f'Steering Vector Layer Sweep  (α={results["alpha"]}, N={results["n_samples"]})\n'
        f'neutral: "{NEUTRAL_PROMPT.strip()}"'
    )
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"layer_sweep.{ext}"), dpi=150)
    plt.close(fig)

# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha",      type=float, default=None)
    parser.add_argument("--n_samples",  type=int,   default=None)
    parser.add_argument("--model_name", type=str,   default=None)
    args = parser.parse_args()

    global ALPHA, N_SAMPLES
    if args.alpha      is not None: ALPHA     = args.alpha
    if args.n_samples  is not None: N_SAMPLES = args.n_samples
    if args.model_name is not None:
        import utils
        utils.MODEL_NAME = args.model_name

    model, tokenizer, num_layers, _ = load_model_and_tokenizer()
    cmu_dict = load_cmu_dict()
    rhymes_with_eep, rhymes_with_est = build_rhyme_checkers(cmu_dict)

    print("Extracting steering vectors...")
    steering_vectors = extract_steering_vector(model, tokenizer, num_layers)
    for ell in [0, num_layers // 4, num_layers // 2, num_layers - 1]:
        print(f"  Layer {ell:2d}: norm={steering_vectors[ell].norm():.4f}")

    results = run(model, tokenizer, steering_vectors, rhymes_with_eep, rhymes_with_est, num_layers)

    out = os.path.join(RESULTS_DIR, "layer_sweep.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out}")

    plot(results)
    print(f"Figures saved to {FIGURES_DIR}/")

    peak = max(results["layer_sweep"], key=lambda r: r["clean_rate"])
    print(f"\nBest layer: {peak['layer']} (clean_rate={peak['clean_rate']:.4f})")
    print(f"→ Use --layer {peak['layer']} in alpha_sweep.py")


if __name__ == "__main__":
    main()
