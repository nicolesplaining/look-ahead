#!/usr/bin/env bash
# Evaluate how often a model produces a rhyming second line.
#
# Override via env vars:
#   MODEL_NAME=Qwen/Qwen3-32B
#   POEMS_PATH=/path/to/poems.jsonl
#   OUTPUT_DIR=/path/to/output
#   MAX_NEW_TOKENS=32
#   MAX_POEMS=600              # default: all
#   TEMPERATURE=0.0            # 0 = greedy; >0 = sampling
#   N_SAMPLES=5                # completions per poem when TEMPERATURE > 0
#   QUANTIZATION=8bit          # "8bit" or "4bit" (requires bitsandbytes)
#   DEVICE=cuda

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# export PYTHONPATH="$PROJECT_ROOT/poem/src:$PROJECT_ROOT/probe/src:$PYTHONPATH"
export PYTHONPATH=""
MODEL_NAME=google/gemma-3-27b-it
POEMS_PATH=$PROJECT_ROOT/poem/data/poems-all-truncated-shuffled.jsonl
OUTPUT_DIR=$PROJECT_ROOT/poem/results/evaluate_rhyming
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
TEMPERATURE=0
N_SAMPLES=1
QUANTIZATION="${QUANTIZATION:-}"   # "8bit" halves bfloat16 memory; "4bit" quarters it
DEVICE="${DEVICE:-cuda}"

MAX_POEMS_ARG=()
[ -n "${MAX_POEMS:-}" ] && MAX_POEMS_ARG=(--max_poems "$MAX_POEMS")

QUANTIZATION_ARG=()
[ -n "$QUANTIZATION" ] && QUANTIZATION_ARG=(--quantization "$QUANTIZATION")

mkdir -p "$OUTPUT_DIR"

echo "Evaluate rhyming"
echo "  model:          $MODEL_NAME"
echo "  poems:          $POEMS_PATH"
echo "  temperature:    $TEMPERATURE"
echo "  n_samples:      $N_SAMPLES"
echo "  max_new_tokens: $MAX_NEW_TOKENS"
echo "  quantization:   ${QUANTIZATION:-none}"
echo "  device:         $DEVICE"
echo "  output_dir:     $OUTPUT_DIR"
echo ""

python -m evaluate_rhyming \
    --model_name     "$MODEL_NAME" \
    --poems_path     "$POEMS_PATH" \
    --output_dir     "$OUTPUT_DIR" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature    "$TEMPERATURE" \
    --n_samples      "$N_SAMPLES" \
    --device         "$DEVICE" \
    "${MAX_POEMS_ARG[@]}" \
    "${QUANTIZATION_ARG[@]}"

echo ""
echo "Done. Results saved to $OUTPUT_DIR/results.json"
