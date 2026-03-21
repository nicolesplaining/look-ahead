#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# ── configurable via env vars or CLI overrides ──────────────────────────────
# MODEL=Qwen/Qwen3-0.6B
# MODEL=google/gemma-3-27b-it
MODEL=meta-llama/Llama-3.2-1B
DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/steering/data/poems-train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/steering/results}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-20}"
LAYERS="${LAYERS:-}"          # space-separated list, e.g. "0 8 16 24 32"; empty = all
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
QUANTIZATION="${QUANTIZATION:-}"   # "8bit" halves bfloat16 memory; "4bit" quarters it (requires bitsandbytes)
PYTHONPATH=""

# Forward extra CLI args (e.g. --layers 0 8 16)
EXTRA_ARGS=("$@")

# ── build --layers flag if set ───────────────────────────────────────────────
LAYERS_FLAG=()
if [ -n "$LAYERS" ]; then
    read -r -a LAYERS_ARR <<< "$LAYERS"
    LAYERS_FLAG=(--layers "${LAYERS_ARR[@]}")
fi

export PYTHONPATH="$PROJECT_ROOT/steering/src:$PYTHONPATH"

QUANTIZATION_FLAG=()
if [ -n "$QUANTIZATION" ]; then
    QUANTIZATION_FLAG=(--quantization "$QUANTIZATION")
fi

python -m steering_probe.compute_vectors \
    --model         "$MODEL" \
    --data-path     "$DATA_PATH" \
    --output-dir    "$OUTPUT_DIR" \
    --context-window "$CONTEXT_WINDOW" \
    --device        "$DEVICE" \
    --dtype         "$DTYPE" \
    "${LAYERS_FLAG[@]}" \
    "${QUANTIZATION_FLAG[@]}" \
    "${EXTRA_ARGS[@]}"
