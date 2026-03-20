#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# ── configurable ─────────────────────────────────────────────────────────────
# MODEL=Qwen/Qwen3-0.6B
# MODEL=google/gemma-3-1b-it
MODEL=meta-llama/Llama-3.2-1B
VECTORS_PATH="${VECTORS_PATH:-$PROJECT_ROOT/steering/results/steering_vectors.pt}"
DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/steering/data/poems-val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/steering/results}"
ALPHA="${ALPHA:-1.5}"
TEMPERATURE=0
N_SAMPLES=1
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-20}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
PYTHONPATH=""

# Optional filters (space-separated lists); leave empty to use all
LAYERS=""           # e.g. "0 8 16 24"
POSITIONS="-1 0" # default: last word + newline token
GEN_POSITIONS="${GEN_POSITIONS:-}"  # generation positions, e.g. "1 2 3"
GEN_VECTOR_POS="${GEN_VECTOR_POS:-0}"
SOURCE="0 1"  # source schemes (half of 10)
TARGET="5 6 7 8 9"  # target schemes (other half) → 10 pairs total

# Forward extra CLI args
EXTRA_ARGS=("$@")

# ── build optional flags ─────────────────────────────────────────────────────
LAYERS_FLAG=()
if [ -n "$LAYERS" ]; then
    read -r -a _arr <<< "$LAYERS"; LAYERS_FLAG=(--layers "${_arr[@]}")
fi

POSITIONS_FLAG=()
if [ -n "$POSITIONS" ]; then
    read -r -a _arr <<< "$POSITIONS"; POSITIONS_FLAG=(--positions "${_arr[@]}")
fi

GEN_POSITIONS_FLAG=()
if [ -n "$GEN_POSITIONS" ]; then
    read -r -a _arr <<< "$GEN_POSITIONS"; GEN_POSITIONS_FLAG=(--gen-positions "${_arr[@]}")
fi

SOURCE_FLAG=()
if [ -n "$SOURCE" ]; then
    read -r -a _arr <<< "$SOURCE"; SOURCE_FLAG=(--source "${_arr[@]}")
fi

TARGET_FLAG=()
if [ -n "$TARGET" ]; then
    read -r -a _arr <<< "$TARGET"; TARGET_FLAG=(--target "${_arr[@]}")
fi

export PYTHONPATH="$PROJECT_ROOT/steering/src:$PYTHONPATH"

python -m steering_probe.run_steering \
    --model          "$MODEL" \
    --vectors-path   "$VECTORS_PATH" \
    --data-path      "$DATA_PATH" \
    --output-dir     "$OUTPUT_DIR" \
    --alpha          "$ALPHA" \
    --temperature    "$TEMPERATURE" \
    --n-samples      "$N_SAMPLES" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --gen-vector-pos "$GEN_VECTOR_POS" \
    --device         "$DEVICE" \
    --dtype          "$DTYPE" \
    "${LAYERS_FLAG[@]}" \
    "${POSITIONS_FLAG[@]}" \
    "${GEN_POSITIONS_FLAG[@]}" \
    "${SOURCE_FLAG[@]}" \
    "${TARGET_FLAG[@]}" \
    "${EXTRA_ARGS[@]}"
