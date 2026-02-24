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

    layers = [r["layer"] for r in results]
    clean_rates = [r["clean_rhyme_rate"] for r in results]
    corrupt_rates = [r["corrupt_rhyme_rate"] for r in results]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax2 = ax.twinx()

    ax2.bar(
        layers,
        clean_rates,
        color="steelblue",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.45,
        label=f"'{clean_word}'-rhyme rate (patched, zoomed)",
    )
    ax2.plot(
        layers,
        clean_rates,
        color="steelblue",
        marker="s",
        markersize=3,
        linewidth=1.0,
    )
    ax.plot(
        layers,
        corrupt_rates,
        color="darkorange",
        marker="o",
        markersize=3,
        linewidth=1.0,
        label=f"'{corrupt_word}'-rhyme rate (patched)",
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

    ax.set_ylabel(f"'{corrupt_word}' rhyme rate")
    clean_axis_max = max(max(clean_rates) * 1.25, 0.01)
    ax2.set_ylim(0, clean_axis_max)
    ax2.set_ylabel(f"'{clean_word}' rhyme rate (zoomed)")
    ax.set_xlabel(f"Layer (patch mode: {patch_mode} @ {patch_label})")
    ax.set_xticks(layers)
    ax.set_xlim(-0.5, max(layers) + 0.5)
    ax.set_title(
        f"Does patching [{patch_label}] transfer the rhyme plan? "
        f"(sampling N={sampling_n} T={sampling_temp})\n"
        f"{model_name} | clean r1='{clean_word}' → corrupt run (r1='{corrupt_word}')"
    )
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    plt.tight_layout()

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = json_path.with_name("patching_results_from_json.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
