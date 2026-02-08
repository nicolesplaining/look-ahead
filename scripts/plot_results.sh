#!/bin/bash
# Plot experiment results from JSON
# Can be run from any directory

set -e

# Get absolute path to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Add src to PYTHONPATH so package is importable
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Path to results JSON
RESULTS_PATH="$PROJECT_ROOT/experiment_results_linear/experiment_results.json"

echo "Plotting results from: $RESULTS_PATH"
echo ""

python -m look_ahead_probe.visualize_results \
    "$RESULTS_PATH" \
    --show-val \
    --show-top5

echo ""
echo "✓ Plots created in $(dirname $RESULTS_PATH)/"
