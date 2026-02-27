#!/bin/bash
# Plot experiment results — overlay multiple result JSONs on one plot per k.
# Can be run from any directory.
#
# Override via env vars:
#   RESULTS_DIR=/path/to/results/dir
#   OUTPUT_DIR=/path/to/output
#   ACC_MIN=0  ACC_MAX=0.8

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

export PYTHONPATH="$PROJECT_ROOT/probe/src:$PYTHONPATH"

RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/probe/results/qwen-3-32B}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_DIR/plots}"
ACC_MIN="${ACC_MIN:-0}"

ACC_MAX=1

mkdir -p "$OUTPUT_DIR"

JSONS=()
LABELS=()
COLORS=()

f="$RESULTS_DIR/experiment_results_linear/experiment_results.json"
if [ -f "$f" ]; then JSONS+=("$f"); LABELS+=("Linear Probe"); COLORS+=("steelblue"); fi

f="$RESULTS_DIR/baselines/bigram/bigram_results.json"
if [ -f "$f" ]; then JSONS+=("$f"); LABELS+=("Bigram"); COLORS+=("orange"); fi

f="$RESULTS_DIR/baselines/unigram/unigram_results.json"
if [ -f "$f" ]; then JSONS+=("$f"); LABELS+=("Unigram"); COLORS+=("gray"); fi

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
    --show-top5 \
    --acc-min "$ACC_MIN" \
    --acc-max "$ACC_MAX" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "✓ Plots saved to $OUTPUT_DIR/"
