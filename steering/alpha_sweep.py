"""
Alpha sweep: at a fixed layer (from layer_sweep.py results), vary the steering
coefficient α and measure "-eep" and "-est" rhyme rates as a function of α.

Usage:
    python alpha_sweep.py --layer 40 [--n_samples 200] [--model_name ...]
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

NEUTRAL_PROMPT     = "He walked out into the open air,\n"

ALPHA_VALUES       = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
N_SAMPLES          = 200
TEMPERATURE        = 0.8
MAX_NEW_TOKENS     = 20
TOP_P              = 0.95

# ── Sweep ───────────────────────────────────────────────────────────────────────

def run(model, tokenizer, steering_vectors, rhymes_with_eep, rhymes_with_est, layer):
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
        "layer":                 layer,
        "n_samples":             N_SAMPLES,
        "temperature":           TEMPERATURE,
        "max_new_tokens":        MAX_NEW_TOKENS,
        "top_p":                 TOP_P,
        "baseline_clean_rate":   b_clean,
        "baseline_corrupt_rate": b_corrupt,
        "baseline_neither_rate": b_neither,
        "alpha_sweep":           [],
    }

    for alpha in tqdm(ALPHA_VALUES, desc="Alpha sweep"):
        completions = run_steered_generation(
            model, tokenizer, NEUTRAL_PROMPT, layer, alpha, steering_vectors,
            N_SAMPLES, TEMPERATURE, MAX_NEW_TOKENS, TOP_P
        )
        clean_rate, corrupt_rate, neither_rate, clean_words, corrupt_words = \
            evaluate_completions(completions, rhymes_with_eep, rhymes_with_est)

        results["alpha_sweep"].append({
            "alpha":         alpha,
            "clean_rate":    clean_rate,
            "corrupt_rate":  corrupt_rate,
            "neither_rate":  neither_rate,
            "clean_words":   clean_words,
            "corrupt_words": corrupt_words,
        })
        print(f"α={alpha:5.2f}  |  clean={clean_rate:.4f}  corrupt={corrupt_rate:.4f}  neither={neither_rate:.4f}")

    return results

# ── Plot ────────────────────────────────────────────────────────────────────────

def plot(results):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    alphas        = [r["alpha"]        for r in results["alpha_sweep"]]
    clean_rates   = [r["clean_rate"]   for r in results["alpha_sweep"]]
    corrupt_rates = [r["corrupt_rate"] for r in results["alpha_sweep"]]
    b_clean   = results["baseline_clean_rate"]
    b_corrupt = results["baseline_corrupt_rate"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(alphas, clean_rates,   color="steelblue", marker="o", linewidth=1.5, label='"-eep" rate (target)')
    ax.plot(alphas, corrupt_rates, color="orange",    marker="o", linewidth=1.5, label='"-est" rate (control)')
    ax.axhline(b_clean,   color="steelblue", linestyle="--", linewidth=1.0, alpha=0.6,
               label=f'Baseline "-eep" = {b_clean:.3f}')
    ax.axhline(b_corrupt, color="orange",    linestyle="--", linewidth=1.0, alpha=0.6,
               label=f'Baseline "-est" = {b_corrupt:.3f}')
    ax.set_xscale("log")
    ax.set_xlabel("α (log scale)")
    ax.set_ylabel("Rhyme Rate")
    ax.set_title(f'Alpha Sweep at Layer {results["layer"]}  (N={results["n_samples"]})')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"alpha_sweep.{ext}"), dpi=150)
    plt.close(fig)

# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer",      type=int, required=True,
                        help="Layer to steer (from layer_sweep.py results)")
    parser.add_argument("--n_samples",  type=int,   default=None)
    parser.add_argument("--model_name", type=str,   default=None)
    args = parser.parse_args()

    global N_SAMPLES
    if args.n_samples  is not None: N_SAMPLES = args.n_samples
    if args.model_name is not None:
        import utils
        utils.MODEL_NAME = args.model_name

    model, tokenizer, num_layers, _ = load_model_and_tokenizer()
    cmu_dict = load_cmu_dict()
    rhymes_with_eep, rhymes_with_est = build_rhyme_checkers(cmu_dict)

    print("Extracting steering vectors...")
    steering_vectors = extract_steering_vector(model, tokenizer, num_layers)

    results = run(model, tokenizer, steering_vectors, rhymes_with_eep, rhymes_with_est, args.layer)

    out = os.path.join(RESULTS_DIR, "alpha_sweep.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out}")

    plot(results)
    print(f"Figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
