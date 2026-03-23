#!/usr/bin/env bash
# Run patch_all_layers_unified.py for every model, one at a time.
#
# Usage:
#   bash run_all_layers_all_models.sh                    # run all (nohup background)
#   bash run_all_layers_all_models.sh Qwen/Qwen3-8B      # run one model (foreground)
#   bash run_all_layers_all_models.sh --worker            # internal use by nohup

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$ROOT_DIR/.venv"
PYTHON="$VENV/bin/python"
EXP_SCRIPT="$SCRIPT_DIR/patch_all_layers_unified.py"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# Models to run (smallest first; 70B last)
MODELS=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
    "Qwen/Qwen3-32B"
    "google/gemma-3-1b-it"
    "google/gemma-3-4b-it"
    "google/gemma-3-12b-it"
    "google/gemma-3-27b-it"
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.1-70B-Instruct"
)

BATCH_SIZE_70B=16

# ── Single-model mode ──────────────────────────────────────────────────────────
if [[ "${1:-}" != "--worker" && -n "${1:-}" ]]; then
    MODEL="$1"
    TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
    SLUG="${MODEL//\//_}"
    LOG="$LOG_DIR/${TIMESTAMP}_all_layers_${SLUG}.log"
    EXTRA_ARGS=""
    if [[ "$MODEL" == *"70B"* ]]; then
        EXTRA_ARGS="--batch-size $BATCH_SIZE_70B"
    fi
    echo "Running: $MODEL $EXTRA_ARGS"
    echo "Log: $LOG"
    "$PYTHON" "$EXP_SCRIPT" --model "$MODEL" $EXTRA_ARGS 2>&1 | tee "$LOG"
    exit 0
fi

# ── Worker mode (sequential, called by nohup) ──────────────────────────────────
if [[ "${1:-}" == "--worker" ]]; then
    TIMESTAMP="${2}"
    echo "Worker started at $(date)"
    for MODEL in "${MODELS[@]}"; do
        SLUG="${MODEL//\//_}"
        LOG="$LOG_DIR/${TIMESTAMP}_all_layers_${SLUG}.log"
        echo
        echo "=== [$MODEL] starting at $(date) ==="
        echo "Log: $LOG"
        EXTRA_ARGS=""
        if [[ "$MODEL" == *"70B"* ]]; then
            EXTRA_ARGS="--batch-size $BATCH_SIZE_70B"
        fi
        "$PYTHON" "$EXP_SCRIPT" --model "$MODEL" $EXTRA_ARGS 2>&1 | tee "$LOG"
        echo "=== [$MODEL] finished at $(date) ==="
    done
    echo
    echo "All models completed at $(date)"
    exit 0
fi

# ── Default: launch all models in background via nohup ────────────────────────
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
MASTER_LOG="$LOG_DIR/nohup_all_layers_master_${TIMESTAMP}.log"

nohup bash "$0" --worker "$TIMESTAMP" > "$MASTER_LOG" 2>&1 &
PID=$!

echo "Started sequential all-layers patching runs in background."
echo "PID: $PID"
echo "Master log: $MASTER_LOG"
echo
echo "Follow progress:"
echo "  tail -f \"$MASTER_LOG\""
echo
echo "Models queued (${#MODELS[@]}):"
for M in "${MODELS[@]}"; do echo "  $M"; done
