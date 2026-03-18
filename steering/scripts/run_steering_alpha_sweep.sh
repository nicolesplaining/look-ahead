#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# ── configurable ─────────────────────────────────────────────────────────────
# MODEL=Qwen/Qwen3-32B
MODEL=google/gemma-3-27b-it
VECTORS_PATH="${VECTORS_PATH:-$PROJECT_ROOT/steering/results/steering_vectors.pt}"
DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/steering/data/poems-val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/steering/results}"
TEMPERATURE=0
N_SAMPLES=1
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-20}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
PYTHONPATH=""

# Alpha values to sweep
ALPHAS=(0.5 1.0 1.5 2.0 2.5 3.0)

# Optional filters (space-separated lists); leave empty to use all
LAYERS=""           # e.g. "0 8 16 24"
POSITIONS="-2 0" # default: last word + newline token
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

# ── sweep ─────────────────────────────────────────────────────────────────────
for ALPHA in "${ALPHAS[@]}"; do
    # Format alpha for filename: replace "." with "_" → e.g. 1.5 → 1_5
    ALPHA_TAG="${ALPHA//./_}"
    TMP_DIR="$OUTPUT_DIR/.alpha_tmp_${ALPHA_TAG}"

    echo "========================================================"
    echo "Running alpha=${ALPHA}  →  results_alpha${ALPHA_TAG}.json"
    echo "========================================================"

    python -m steering_probe.run_steering \
        --model          "$MODEL" \
        --vectors-path   "$VECTORS_PATH" \
        --data-path      "$DATA_PATH" \
        --output-dir     "$TMP_DIR" \
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

    mv "$TMP_DIR/results.json" "$OUTPUT_DIR/results_alpha${ALPHA_TAG}.json"
    rm -rf "$TMP_DIR"

    echo "Saved → $OUTPUT_DIR/results_alpha${ALPHA_TAG}.json"
done

echo ""
echo "Alpha sweep complete. Results in $OUTPUT_DIR:"
ls "$OUTPUT_DIR"/results_alpha*.json
