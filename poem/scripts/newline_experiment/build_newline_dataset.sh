#!/bin/bash
# Build newline_experiment dataset from existing activations .pt files.
# Extracts i=0 activations and computes targets for k=0..max_k.
# Requires a tokenizer (model must be accessible via HuggingFace).
#
# Override via env vars:
#   MODEL_NAME=Qwen/Qwen2.5-7B
#   DATA_DIR=/path/to/existing/activations
#   OUT_DIR=/path/to/output
#   MAX_K=5

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

export PYTHONPATH="$PROJECT_ROOT/poem/src:$PROJECT_ROOT/probe/src:$PYTHONPATH"

MODEL_NAME=Qwen/Qwen3-32B
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/poem/data}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/poem/data}"
MAX_K="${MAX_K:-5}"

echo "Building newline_experiment datasets"
echo "  model:   $MODEL_NAME"
echo "  data:    $DATA_DIR"
echo "  output:  $OUT_DIR"
echo "  max_k:   $MAX_K"
echo ""

for SPLIT in train val; do
    INPUT="$DATA_DIR/activations_${SPLIT}.pt"
    OUTPUT="$OUT_DIR/newline_${SPLIT}.pt"
    if [ ! -f "$INPUT" ]; then
        echo "  (skipping $SPLIT: $INPUT not found)"
        continue
    fi
    echo "── $SPLIT ──"
    python -m newline_experiment.build_newline_dataset \
        --input      "$INPUT" \
        --output     "$OUTPUT" \
        --model_name "$MODEL_NAME" \
        --max_k      "$MAX_K"
    echo ""
done

echo "✓ Done."
