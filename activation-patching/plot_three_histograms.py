#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def make_plot(json_path: Path, output_path: Path, swap_labels: bool = False) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    results = data["results"]
    layers = [r["layer"] for r in results]

    clean_rates = [r["clean_rhyme_rate"] for r in results]
    corrupt_rates = [r["corrupt_rhyme_rate"] for r in results]
    no_rhyme_rates = [max(0.0, 1.0 - c - k) for c, k in zip(clean_rates, corrupt_rates)]

    baseline = data.get("baseline", {})
    baseline_clean = baseline.get("unpatched_corrupt_clean_rhyme_rate", 0.0)
    baseline_corrupt = baseline.get("unpatched_corrupt_corrupt_rhyme_rate", 0.0)
    baseline_no = max(0.0, 1.0 - baseline_clean - baseline_corrupt)

    clean_word = data.get("clean_rhyme_word", "clean")
    corrupt_word = data.get("corrupt_rhyme_word", "corrupt")
    model_name = data.get("model_name", "model")
    patch_mode = data.get("patch_mode", "?")
    patch_label = data.get("patch_label", "?")
    sampling_n = data.get("sampling_n")
    sampling_temp = data.get("sampling_temp")

    top_label_word = corrupt_word if swap_labels else clean_word
    mid_label_word = clean_word if swap_labels else corrupt_word
    suffix = "" if swap_labels else ""

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].bar(
        layers,
        clean_rates,
        color="steelblue",
        edgecolor="white",
        linewidth=0.5,
    )
    axes[0].axhline(
        baseline_clean,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Baseline rate = {baseline_clean:.3f}",
    )
    axes[0].set_ylabel("Rate")
    axes[0].set_title(f"Patched '{top_label_word}' rhyme rate{suffix}")
    axes[0].legend(loc="upper right")

    axes[1].bar(
        layers,
        corrupt_rates,
        color="darkorange",
        edgecolor="white",
        linewidth=0.5,
    )
    axes[1].axhline(
        baseline_corrupt,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=f"Baseline rate = {baseline_corrupt:.3f}",
    )
    axes[1].set_ylabel("Rate")
    axes[1].set_title(f"Patched '{mid_label_word}' rhyme rate{suffix}")
    axes[1].legend(loc="upper right")

    axes[2].bar(
        layers,
        no_rhyme_rates,
        color="gray",
        edgecolor="white",
        linewidth=0.5,
    )
    axes[2].axhline(
        baseline_no,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Baseline rate = {baseline_no:.3f}",
    )
    axes[2].set_ylabel("Rate")
    axes[2].set_title("Patched no-rhyme rate")
    axes[2].legend(loc="upper right")

    for ax in axes:
        ax.set_xlim(-0.5, max(layers) + 0.5)
        ax.set_xticks(layers)

    axes[2].set_xlabel("Layer")
    fig.suptitle("Activation Patching", y=0.995, fontweight="bold", ha="center")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 3 histogram plot from generations.json")
    parser.add_argument("json_path", help="Path to generations.json")
    parser.add_argument(
        "--output",
        help="Optional output image path. Default: patching_results_newline_sampling_histograms_custom.png next to JSON",
    )
    parser.add_argument(
        "--swap-labels",
        action="store_true",
        help="Swap displayed clean/corrupt labels while keeping underlying data the same.",
    )
    args = parser.parse_args()

    json_path = Path(args.json_path).expanduser().resolve()
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = json_path.with_name("patching_results_newline_sampling_histograms_custom.png")

    make_plot(json_path=json_path, output_path=output_path, swap_labels=args.swap_labels)


if __name__ == "__main__":
    main()
