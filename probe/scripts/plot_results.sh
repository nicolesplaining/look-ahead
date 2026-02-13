#!/bin/bash
# Plot experiment results from JSON
# Can be run from any directory

set -e

# Get absolute path to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Add src to PYTHONPATH so package is importable
export PYTHONPATH="$PROJECT_ROOT/probe/src:$PYTHONPATH"

# Path to results JSON
# RESULTS_PATH="$PROJECT_ROOT/probe/results/qwen-3-32B/experiment_results_linear/experiment_results.json"
RESULTS_PATH="$PROJECT_ROOT/probe/results/baselines/trigram/trigram_results.json"

# Y-axis range for accuracy plots (override via env vars, e.g. ACC_MAX=0.5)
ACC_MIN=0
ACC_MAX=0.5

echo "Plotting results from: $RESULTS_PATH"
echo "Accuracy y-axis: [$ACC_MIN, $ACC_MAX]"
echo ""

# python -m look_ahead_probe.visualize_results \
#     "$RESULTS_PATH" \
#     --show-val \
#     --show-top5 \
#     --acc-min "$ACC_MIN" \
#     --acc-max "$ACC_MAX"

python -m look_ahead_probe.visualize_results \
    "$RESULTS_PATH" \
    --show-val \
    --acc-min "$ACC_MIN" \
    --acc-max "$ACC_MAX"


echo ""
echo "✓ Plots created in $(dirname $RESULTS_PATH)/"
