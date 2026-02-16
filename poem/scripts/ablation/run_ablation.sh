#!/bin/bash
# Evaluate how often a model produces a rhyming second line.
# Runs both modes (with_newline and without_newline) in one model load.
#
# Override via env vars:
#   MODEL_NAME=Qwen/Qwen3-32B
#   POEMS_PATH=/path/to/poems.jsonl
#   MODE=both                  # {with_newline, without_newline, both}
#   OUTPUT_DIR=/path/to/output
#   MAX_NEW_TOKENS=16
#   MAX_POEMS=600              # default: all

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

export PYTHONPATH="$PROJECT_ROOT/poem/src:$PROJECT_ROOT/probe/src:$PYTHONPATH"

MODEL_NAME=Qwen/Qwen2.5-72B
POEMS_PATH="${POEMS_PATH:-$PROJECT_ROOT/poem/data/poems-original-truncated-shuffled.jsonl}"
MODE="${MODE:-both}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/poem/results/ablation/qwen2.5-72B}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"

MAX_POEMS_ARG=()
[ -n "$MAX_POEMS" ] && MAX_POEMS_ARG=(--max_poems "$MAX_POEMS")

mkdir -p "$OUTPUT_DIR"

echo "Ablation: rhyming evaluation"
echo "  model:          $MODEL_NAME"
echo "  poems:          $POEMS_PATH"
echo "  mode:           $MODE"
echo "  max_new_tokens: $MAX_NEW_TOKENS"
echo "  output_dir:     $OUTPUT_DIR"
echo ""

python -m ablation.evaluate_rhyming \
    --model_name     "$MODEL_NAME" \
    --poems_path     "$POEMS_PATH" \
    --mode           "$MODE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --output_dir     "$OUTPUT_DIR" \
    "${MAX_POEMS_ARG[@]}"

echo ""
echo "✓ Done. Results saved to $OUTPUT_DIR/"
