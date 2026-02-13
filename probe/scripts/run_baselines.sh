#!/bin/bash
# Evaluate n-gram baselines and write results JSONs compatible with plot_results.sh.
# Does NOT require a GPU; only the tokenizer is loaded.
# Can be run from any directory.
#
# Override defaults via env vars, e.g.:
#   MODEL_NAME=Qwen/Qwen2.5-7B bash probe/scripts/run_baselines.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

export PYTHONPATH="$PROJECT_ROOT/probe/src:$PYTHONPATH"

MODEL_NAME="${MODEL_NAME:-}"  # Only needed for .texts.jsonl or .pt inputs

# Prefer .tokens.jsonl (exact token IDs, no re-tokenization) > .texts.jsonl > .pt
if [ -f "$PROJECT_ROOT/probe/data/activations_train.tokens.jsonl" ]; then
    TRAIN_DATASET="${TRAIN_DATASET:-$PROJECT_ROOT/probe/data/activations_train.tokens.jsonl}"
    VAL_DATASET="${VAL_DATASET:-$PROJECT_ROOT/probe/data/activations_val.tokens.jsonl}"
elif [ -f "$PROJECT_ROOT/probe/data/activations_train.texts.jsonl" ]; then
    TRAIN_DATASET="${TRAIN_DATASET:-$PROJECT_ROOT/probe/data/activations_train.texts.jsonl}"
    VAL_DATASET="${VAL_DATASET:-$PROJECT_ROOT/probe/data/activations_val.texts.jsonl}"
else
    TRAIN_DATASET="${TRAIN_DATASET:-$PROJECT_ROOT/probe/data/activations_train.pt}"
    VAL_DATASET="${VAL_DATASET:-$PROJECT_ROOT/probe/data/activations_val.pt}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/probe/results/baselines}"
MAX_K="${MAX_K:-3}"

if [ ! -f "$TRAIN_DATASET" ]; then
    echo "ERROR: Training dataset not found at $TRAIN_DATASET"
    echo "Run build_dataset.sh first."
    exit 1
fi

if [ ! -f "$VAL_DATASET" ]; then
    echo "ERROR: Validation dataset not found at $VAL_DATASET"
    exit 1
fi

echo "Running n-gram baselines"
echo "Train dataset: $TRAIN_DATASET"
echo "Val dataset:   $VAL_DATASET"
echo "Output dir:    $OUTPUT_DIR"
echo "max_k:         $MAX_K"
echo ""

# Build optional --model_name arg (only needed for .texts.jsonl / .pt inputs)
MODEL_ARG=""
if [ -n "$MODEL_NAME" ]; then
    MODEL_ARG="--model_name $MODEL_NAME"
fi

python -m look_ahead_probe.baseline \
    --train_dataset "$TRAIN_DATASET" \
    --val_dataset   "$VAL_DATASET" \
    $MODEL_ARG \
    --output_dir    "$OUTPUT_DIR" \
    --max_k         "$MAX_K"

echo ""
echo "✓ Baseline results written to $OUTPUT_DIR/"
echo ""
echo "To plot alongside probe results, run e.g.:"
echo "  RESULTS_PATH=$OUTPUT_DIR/unigram_results.json bash probe/scripts/plot_results.sh"
echo "  RESULTS_PATH=$OUTPUT_DIR/bigram_results.json  bash probe/scripts/plot_results.sh"
echo "  RESULTS_PATH=$OUTPUT_DIR/trigram_results.json bash probe/scripts/plot_results.sh"
