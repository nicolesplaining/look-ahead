#!/usr/bin/env bash
# Master script: chains all 3 steps of the steering experiment.
# Launch this and go to sleep — results will be in steering/results/ when done.
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

# Use /workspace for HF model cache (root volume is tiny on RunPod)
export HF_HOME="${HF_HOME:-/workspace/tmp/hf_cache}"
mkdir -p "$HF_HOME"

echo "============================================"
echo "  Steering Experiment — Full Pipeline"
echo "  $(date)"
echo "============================================"

# Step 1: Compute steering vectors (~30-60 min on 1xH100)
echo ""
echo ">>> Step 1/3: Computing steering vectors ..."
echo "    Started at $(date)"
bash steering/scripts/compute_vectors.sh
echo "    Finished at $(date)"

# Step 2: Run steering on all pairs (~several hours for 90 pairs x layers x positions)
echo ""
echo ">>> Step 2/3: Running steering experiment ..."
echo "    Started at $(date)"
bash steering/scripts/run_steering.sh
echo "    Finished at $(date)"

# Step 3: Plot results
echo ""
echo ">>> Step 3/3: Plotting results ..."
echo "    Started at $(date)"
bash steering/scripts/plot_results.sh
echo "    Finished at $(date)"

echo ""
echo "============================================"
echo "  DONE — check steering/results/"
echo "  Completed at $(date)"
echo "============================================"
