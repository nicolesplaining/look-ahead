#!/bin/bash
# Plot experiment results — single plot with one line per k value.
# Can be run from any directory.
#
# Override via env vars:
#   EXPERIMENT_JSON=/path/to/experiment_results.json
#   UNIGRAM_JSON=/path/to/unigram_results.json   (set to "" to disable)
#   OUTPUT_DIR=/path/to/output
#   K_VALUES="1 2 3 8"   (space-separated; default: all k values)
#   ACC_MIN=0  ACC_MAX=0.65

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

export PYTHONPATH="$PROJECT_ROOT/probe/src:$PYTHONPATH"

RESULTS_DIR=$PROJECT_ROOT/probe/results/Llama-3.1-70B-Instruct
EXPERIMENT_JSON="${EXPERIMENT_JSON:-$RESULTS_DIR/experiment_results_linear/experiment_results.json}"
UNIGRAM_JSON="${UNIGRAM_JSON:-$RESULTS_DIR/baselines/unigram_results.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_DIR/plots}"
ACC_MIN="${ACC_MIN:-0}"
ACC_MAX=0.7
K_VALUES="${K_VALUES:-1 2 3 8}"   # empty = all k values

mkdir -p "$OUTPUT_DIR"

if [ ! -f "$EXPERIMENT_JSON" ]; then
    echo "ERROR: experiment JSON not found: $EXPERIMENT_JSON"
    echo "Set EXPERIMENT_JSON= or run train_probes.sh first."
    exit 1
fi

echo "Experiment JSON : $EXPERIMENT_JSON"

UNIGRAM_ARG=()
if [ -n "$UNIGRAM_JSON" ] && [ -f "$UNIGRAM_JSON" ]; then
    echo "Unigram baseline: $UNIGRAM_JSON"
    UNIGRAM_ARG=(--unigram-json "$UNIGRAM_JSON")
else
    echo "Unigram baseline: (none)"
fi

echo "Output dir      : $OUTPUT_DIR"
echo "Accuracy y-axis : [$ACC_MIN, $ACC_MAX]"
echo ""

K_VALS_ARG=()
[ -n "$K_VALUES" ] && K_VALS_ARG=(--k-values $K_VALUES)

# Decreasing shades of blue for k=1,2,3,8 (dark → light)
BLUE_GRADIENT=("#2676AD" "#4195C3" "#6AAFD4" "#93C4E0")

COMMON_ARGS=(
    "$EXPERIMENT_JSON"
    --single-plot
    "${UNIGRAM_ARG[@]}"
    --colors "${BLUE_GRADIENT[@]}"
    --acc-min "$ACC_MIN"
    --acc-max "$ACC_MAX"
    --output-dir "$OUTPUT_DIR"
    "${K_VALS_ARG[@]}"
)

python -m look_ahead_probe.visualize_results "${COMMON_ARGS[@]}" --show-val   --file-name val_accuracy.png
python -m look_ahead_probe.visualize_results "${COMMON_ARGS[@]}" --show-top5  --file-name top5_accuracy.png

echo ""
echo "Plots saved to $OUTPUT_DIR/"
