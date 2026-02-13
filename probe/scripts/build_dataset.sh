#!/bin/bash
# Build activation datasets (Step 1 of decoupled pipeline).
# Run on a big GPU once; the .pt outputs can be reused for repeated training runs.
# Can be run from any directory.
#
# Override defaults via env vars, e.g.:
#   MODEL_NAME=gpt2 MAX_TRAIN_PROMPTS=50 bash probe/scripts/build_dataset.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

export PYTHONPATH="$PROJECT_ROOT/probe/src:$PYTHONPATH"

MODEL_NAME="Qwen/Qwen2.5-32B"
MAX_K=3
MAX_NEW_TOKENS=32
MAX_TRAIN_PROMPTS=100
MAX_VAL_PROMPTS=10
DEVICE="${DEVICE:-cuda}"

TRAIN_INPUT="$PROJECT_ROOT/probe/data/train-pile.jsonl"
VAL_INPUT="$PROJECT_ROOT/probe/data/val-pile.jsonl"
TRAIN_OUTPUT="$PROJECT_ROOT/probe/data/activations_train.pt"
VAL_OUTPUT="$PROJECT_ROOT/probe/data/activations_val.pt"

if [ ! -f "$TRAIN_INPUT" ]; then
    echo "ERROR: Training data not found at $TRAIN_INPUT"
    exit 1
fi

echo "Building activation datasets from: $PROJECT_ROOT"
echo "Model:  $MODEL_NAME"
echo "max_k:  $MAX_K"
echo "Device: $DEVICE"
echo ""

# --- Step 1a: training activations ---
echo "=== Step 1a: Building training activations ==="
TRAIN_CMD=(
    python -m look_ahead_probe.build_look_ahead_activation_dataset
    --model_name "$MODEL_NAME"
    --prompts_path "$TRAIN_INPUT"
    --max_k "$MAX_K"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --output_path "$TRAIN_OUTPUT"
    --device "$DEVICE"
)
if [ -n "$MAX_TRAIN_PROMPTS" ]; then
    TRAIN_CMD+=(--max_prompts "$MAX_TRAIN_PROMPTS")
fi
"${TRAIN_CMD[@]}"

# --- Step 1b: validation activations (skip if val file absent) ---
if [ -f "$VAL_INPUT" ]; then
    echo ""
    echo "=== Step 1b: Building validation activations ==="
    VAL_CMD=(
        python -m look_ahead_probe.build_look_ahead_activation_dataset
        --model_name "$MODEL_NAME"
        --prompts_path "$VAL_INPUT"
        --max_k "$MAX_K"
        --max_new_tokens "$MAX_NEW_TOKENS"
        --output_path "$VAL_OUTPUT"
        --device "$DEVICE"
    )
    if [ -n "$MAX_VAL_PROMPTS" ]; then
        VAL_CMD+=(--max_prompts "$MAX_VAL_PROMPTS")
    fi
    "${VAL_CMD[@]}"
else
    echo ""
    echo "(Skipping validation: $VAL_INPUT not found)"
fi

echo ""
echo "✓ Activation datasets saved to $PROJECT_ROOT/probe/data/"
