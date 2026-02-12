#!/bin/bash
# Plot poem probe results (Step 3 of decoupled pipeline).
# Can be run from any directory.
#
# To use a different results file, set RESULTS_PATH, e.g.:
#   RESULTS_PATH=/path/to/experiment_results.json bash poem/scripts/plot_results.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Only probe/src needed (visualize_results lives in look_ahead_probe)
export PYTHONPATH="$PROJECT_ROOT/probe/src:$PYTHONPATH"

RESULTS_PATH="${RESULTS_PATH:-$PROJECT_ROOT/poem/results/experiment_results_linear/experiment_results.json}"

if [ ! -f "$RESULTS_PATH" ]; then
    echo "ERROR: Results file not found at $RESULTS_PATH"
    echo "Run train_probes.sh first, or set RESULTS_PATH."
    exit 1
fi

echo "Plotting poem results from: $RESULTS_PATH"
echo ""

python -m look_ahead_probe.visualize_results \
    "$RESULTS_PATH" \
    --show-val \
    --show-top5

echo ""
echo "✓ Plots created in $(dirname "$RESULTS_PATH")/"
