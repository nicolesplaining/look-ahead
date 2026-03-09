#!/usr/bin/env bash
# Normalized steering experiment — reuses existing steering vectors.
# Launch: nohup bash steering/scripts/run_all_normalized.sh > steering/results/normalized/run.log 2>&1 &
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

# Use /workspace for HF model cache (root volume is tiny on RunPod)
export HF_HOME="${HF_HOME:-/workspace/tmp/hf_cache}"
mkdir -p "$HF_HOME"

OUTPUT_DIR="$PROJECT_ROOT/steering/results/normalized"
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "  Normalized Steering Experiment"
echo "  $(date)"
echo "============================================"

# Step 1: Run normalized steering (reuses existing steering_vectors.pt)
echo ""
echo ">>> Step 1/2: Running normalized steering ..."
echo "    Started at $(date)"
OUTPUT_DIR="$OUTPUT_DIR" bash steering/scripts/run_steering_normalized.sh
echo "    Finished at $(date)"

# Step 2: Plot results
echo ""
echo ">>> Step 2/2: Plotting results ..."
echo "    Started at $(date)"
RESULTS_PATH="$OUTPUT_DIR/results.json" OUTPUT_DIR="$OUTPUT_DIR/plots" bash steering/scripts/plot_results.sh
echo "    Finished at $(date)"

echo ""
echo "============================================"
echo "  DONE — check steering/results/normalized/"
echo "  Completed at $(date)"
echo "============================================"
