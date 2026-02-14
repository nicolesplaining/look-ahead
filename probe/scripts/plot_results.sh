#!/bin/bash
# Plot experiment results — overlay multiple result JSONs on one plot per k.
# Can be run from any directory.
#
# Usage examples:
#
#   # Single result
#   bash probe/scripts/plot_results.sh
#
#   # Override paths via env vars
#   PROBE_RESULTS=/path/to/experiment_results.json bash probe/scripts/plot_results.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

export PYTHONPATH="$PROJECT_ROOT/probe/src:$PYTHONPATH"

# ------------------------------------------------------------------
# Paths to result JSONs (edit or override via env vars)
# ------------------------------------------------------------------
RESULTS_DIR="$PROJECT_ROOT/probe/results/qwen-3-32B"

PROBE_RESULTS="${PROBE_RESULTS:-$RESULTS_DIR/experiment_results_linear/experiment_results.json}"
UNIGRAM_RESULTS="${UNIGRAM_RESULTS:-$RESULTS_DIR/baselines/unigram/unigram_results.json}"
BIGRAM_RESULTS="${BIGRAM_RESULTS:-$RESULTS_DIR/baselines/bigram/bigram_results.json}"
TRIGRAM_RESULTS="${TRIGRAM_RESULTS:-$RESULTS_DIR/baselines/trigram/trigram_results.json}"

OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_DIR/plots}"
ACC_MIN="${ACC_MIN:-0}"
ACC_MAX="${ACC_MAX:-0.5}"

mkdir -p "$OUTPUT_DIR"

# ------------------------------------------------------------------
# Build the argument list — only include files that exist
# ------------------------------------------------------------------
JSONS=()
LABELS=()
COLORS=()

if [ -f "$PROBE_RESULTS" ]; then
    JSONS+=("$PROBE_RESULTS")
    LABELS+=("Linear Probe")
    COLORS+=("steelblue")
fi

if [ -f "$TRIGRAM_RESULTS" ]; then
    JSONS+=("$TRIGRAM_RESULTS")
    LABELS+=("Trigram")
    COLORS+=("tomato")
fi

if [ -f "$BIGRAM_RESULTS" ]; then
    JSONS+=("$BIGRAM_RESULTS")
    LABELS+=("Bigram")
    COLORS+=("orange")
fi

if [ -f "$UNIGRAM_RESULTS" ]; then
    JSONS+=("$UNIGRAM_RESULTS")
    LABELS+=("Unigram")
    COLORS+=("gray")
fi

if [ ${#JSONS[@]} -eq 0 ]; then
    echo "ERROR: No result JSONs found. Run train_probes.sh and/or run_baselines.sh first."
    exit 1
fi

echo "Plotting ${#JSONS[@]} result(s) → $OUTPUT_DIR"
echo "Accuracy y-axis: [$ACC_MIN, $ACC_MAX]"
echo ""

python -m look_ahead_probe.visualize_results \
    "${JSONS[@]}" \
    --labels "${LABELS[@]}" \
    --colors "${COLORS[@]}" \
    --show-val \
    --acc-min "$ACC_MIN" \
    --acc-max "$ACC_MAX" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "✓ Plots saved to $OUTPUT_DIR/"
