#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate patching plot from an existing generations.json file."
    )
    parser.add_argument("json_path", help="Path to generations.json")
    parser.add_argument(
        "--output",
        help="Optional output image path (default: sibling PNG next to JSON)",
    )
    args = parser.parse_args()

    json_path = Path(args.json_path).expanduser().resolve()
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    if not results:
        raise ValueError("No results found in JSON.")

    sampling_mode = data.get("sampling_mode", False)
    if not sampling_mode:
        raise ValueError("This plotting script currently supports sampling_mode=true runs.")

    clean_word = data.get("clean_rhyme_word", "clean")
    corrupt_word = data.get("corrupt_rhyme_word", "corrupt")
    patch_mode = data.get("patch_mode", "unknown")
    patch_label = data.get("patch_label", "unknown")
    model_name = data.get("model_name", "model")
    sampling_n = data.get("sampling_n")
    sampling_temp = data.get("sampling_temp")

    baseline = data.get("baseline", {})
    baseline_clean_rate = baseline.get("unpatched_corrupt_clean_rhyme_rate", 0.0)
    baseline_corrupt_rate = baseline.get("unpatched_corrupt_corrupt_rhyme_rate", 0.0)
    baseline_no_rhyme_rate = max(0.0, 1.0 - baseline_clean_rate - baseline_corrupt_rate)

    layers = [r["layer"] for r in results]
    clean_rates = [r["clean_rhyme_rate"] for r in results]
    corrupt_rates = [r["corrupt_rhyme_rate"] for r in results]
    no_rhyme_rates = [max(0.0, 1.0 - clean - corrupt) for clean, corrupt in zip(clean_rates, corrupt_rates)]

    fig, (ax_clean, ax_corrupt, ax_no_rhyme) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax_clean.bar(
        layers,
        clean_rates,
        color="steelblue",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )
    ax_clean.axhline(
        baseline_clean_rate,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"baseline ({baseline_clean_rate:.3f})",
    )
    ax_clean.set_ylabel(f"'{clean_word}' rhyme")
    ax_clean.set_title(f"Clean rhyme ('{clean_word}')")
    ax_clean.set_ylim(0, max(max(clean_rates), baseline_clean_rate) * 1.2 + 0.01)
    ax_clean.legend(loc="upper right")

    ax_corrupt.bar(
        layers,
        corrupt_rates,
        color="darkorange",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )
    ax_corrupt.axhline(
        baseline_corrupt_rate,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=f"baseline ({baseline_corrupt_rate:.3f})",
    )
    ax_corrupt.set_ylabel(f"'{corrupt_word}' rhyme")
    ax_corrupt.set_title(f"Corrupt rhyme ('{corrupt_word}')")
    ax_corrupt.set_ylim(0, max(max(corrupt_rates), baseline_corrupt_rate) * 1.2 + 0.01)
    ax_corrupt.legend(loc="upper right")

    ax_no_rhyme.bar(
        layers,
        no_rhyme_rates,
        color="slategray",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )
    ax_no_rhyme.axhline(
        baseline_no_rhyme_rate,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"baseline ({baseline_no_rhyme_rate:.3f})",
    )
    ax_no_rhyme.set_ylabel("no rhyme")
    ax_no_rhyme.set_title("No rhyme")
    ax_no_rhyme.set_ylim(0, max(max(no_rhyme_rates), baseline_no_rhyme_rate) * 1.2 + 0.01)
    ax_no_rhyme.legend(loc="upper right")

    ax_no_rhyme.set_xlabel(f"Layer (patch mode: {patch_mode} @ {patch_label})")
    ax_no_rhyme.set_xticks(layers)
    ax_no_rhyme.set_xlim(-0.5, max(layers) + 0.5)

    fig.suptitle(
        f"Does patching [{patch_label}] transfer the rhyme plan? "
        f"(sampling N={sampling_n} T={sampling_temp})\n"
        f"{model_name} | clean r1='{clean_word}' → corrupt run (r1='{corrupt_word}')"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = json_path.with_name("patching_results_from_json.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
