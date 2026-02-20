#!/bin/bash
# Build poem activation datasets (Step 1 of decoupled pipeline).
# Run on a big GPU once; the .pt outputs can be reused for repeated training runs.
# Can be run from any directory.
#
# Override defaults via env vars, e.g.:
#   MODEL_NAME=gpt2 MAX_TRAIN_PROMPTS=400 MAX_VAL_PROMPTS=100 bash poem/scripts/build_dataset.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# poem_probe imports look_ahead_probe, so both src dirs are needed
export PYTHONPATH="$PROJECT_ROOT/poem/src:$PROJECT_ROOT/probe/src:$PYTHONPATH"

MODEL_NAME=Qwen/Qwen3-32B
MAX_BACK="${MAX_BACK:-8}"          # tokens before the first-line \n to store (i = -1 ... -MAX_BACK)
MAX_NEW_TOKENS=16
MAX_TRAIN_PROMPTS="${MAX_TRAIN_PROMPTS:-}"
MAX_VAL_PROMPTS="${MAX_VAL_PROMPTS:-}"
DEVICE="${DEVICE:-cuda}"

TRAIN_INPUT="$PROJECT_ROOT/poem/data/poems-all-train.jsonl"
VAL_INPUT="$PROJECT_ROOT/poem/data/poems-all-val.jsonl"
TRAIN_OUTPUT="$PROJECT_ROOT/poem/data/activations_train.pt"
VAL_OUTPUT="$PROJECT_ROOT/poem/data/activations_val.pt"

if [ ! -f "$TRAIN_INPUT" ]; then
    echo "ERROR: Training poems not found at $TRAIN_INPUT"
    exit 1
fi

echo "Building poem activation datasets from: $PROJECT_ROOT"
echo "Model:          $MODEL_NAME"
echo "max_back:       $MAX_BACK"
echo "max_new_tokens: $MAX_NEW_TOKENS"
echo "Device:         $DEVICE"
echo ""

# --- Step 1a: training activations ---
echo "=== Step 1a: Building training activations ==="
TRAIN_CMD=(
    python -m extract_poem_dataset
    --model_name "$MODEL_NAME"
    --poems_path "$TRAIN_INPUT"
    --output_path "$TRAIN_OUTPUT"
    --max_back "$MAX_BACK"
    --max_new_tokens "$MAX_NEW_TOKENS"
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
        python -m extract_poem_dataset
        --model_name "$MODEL_NAME"
        --poems_path "$VAL_INPUT"
        --output_path "$VAL_OUTPUT"
        --max_back "$MAX_BACK"
        --max_new_tokens "$MAX_NEW_TOKENS"
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
echo "✓ Poem activation datasets saved to $PROJECT_ROOT/poem/data/"
