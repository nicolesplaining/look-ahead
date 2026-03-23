#!/usr/bin/env bash
# Run aggregate_experiment_unified.py for every untested model.
# Already done (N=20): Qwen/Qwen3-32B, google/gemma-3-27b-it
# This script runs N=100 for all remaining models, one at a time.
#
# Usage:
#   bash run_aggregate_all_models.sh                    # run all
#   bash run_aggregate_all_models.sh Qwen/Qwen3-8B      # run one model
#   bash run_aggregate_all_models.sh --worker            # internal use by nohup

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$ROOT_DIR/.venv"
PYTHON="$VENV/bin/python"
EXP_SCRIPT="$SCRIPT_DIR/aggregate_experiment_unified.py"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# Models to run (in rough order of size, smallest first)
MODELS=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
    "google/gemma-3-1b-it"
    "google/gemma-3-4b-it"
    "google/gemma-3-12b-it"
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Llama-3.1-8B"
    "meta-llama/Llama-3.3-70B-Instruct"
    "meta-llama/Llama-3.1-70B"
    "meta-llama/Llama-3.1-70B-Instruct"
    # Already done at N=20; re-run at N=100 if desired by uncommenting:
    # "Qwen/Qwen3-32B"
    # "google/gemma-3-27b-it"
)

# ── Single-model mode ──────────────────────────────────────────────────────────
if [[ "${1:-}" != "--worker" && -n "${1:-}" ]]; then
    MODEL="$1"
    TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
    SLUG="${MODEL//\//_}"
    LOG="$LOG_DIR/${TIMESTAMP}_${SLUG}.log"
    echo "Running: $MODEL"
    echo "Log: $LOG"
    "$PYTHON" "$EXP_SCRIPT" --model "$MODEL" 2>&1 | tee "$LOG"
    exit 0
fi

# ── Worker mode (sequential, called by nohup) ──────────────────────────────────
if [[ "${1:-}" == "--worker" ]]; then
    TIMESTAMP="${2}"
    echo "Worker started at $(date)"
    for MODEL in "${MODELS[@]}"; do
        SLUG="${MODEL//\//_}"
        LOG="$LOG_DIR/${TIMESTAMP}_${SLUG}.log"
        echo
        echo "=== [$MODEL] starting at $(date) ==="
        echo "Log: $LOG"
        "$PYTHON" "$EXP_SCRIPT" --model "$MODEL" 2>&1 | tee "$LOG"
        echo "=== [$MODEL] finished at $(date) ==="
    done
    echo
    echo "All models completed at $(date)"
    exit 0
fi

# ── Default: launch all models in background via nohup ────────────────────────
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
MASTER_LOG="$LOG_DIR/nohup_master_${TIMESTAMP}.log"

nohup bash "$0" --worker "$TIMESTAMP" > "$MASTER_LOG" 2>&1 &
PID=$!

echo "Started sequential aggregate patching runs in background."
echo "PID: $PID"
echo "Master log: $MASTER_LOG"
echo
echo "Follow progress:"
echo "  tail -f \"$MASTER_LOG\""
echo
echo "Models queued (${#MODELS[@]}):"
for M in "${MODELS[@]}"; do echo "  $M"; done
