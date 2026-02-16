#!/bin/bash
# Train newline_experiment probes (fixed i=0 activation, varying target k).
# Reads newline_{train,val}.pt; writes experiment_results.json.
#
# Override via env vars:
#   DATA_DIR=/path/to/newline datasets
#   OUTPUT_DIR=/path/to/results
#   MAX_K=5
#   NUM_EPOCHS=10

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

export PYTHONPATH="$PROJECT_ROOT/poem/src:$PROJECT_ROOT/probe/src:$PYTHONPATH"

DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/poem/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/poem/results/newline_experiment}"
MAX_K="${MAX_K:-5}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
BATCH_SIZE=32
LEARNING_RATE=2e-4
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-3}"
DEVICE="${DEVICE:-cuda}"

TRAIN_PT="$DATA_DIR/newline_train.pt"
VAL_PT="$DATA_DIR/newline_val.pt"

if [ ! -f "$TRAIN_PT" ]; then
    echo "ERROR: $TRAIN_PT not found. Run build_newline_dataset.sh first."
    exit 1
fi

VAL_ARG=()
[ -f "$VAL_PT" ] && VAL_ARG=(--val_dataset "$VAL_PT")

echo "Training newline_experiment probes"
echo "  train:      $TRAIN_PT"
echo "  val:        ${VAL_PT} ($([ -f "$VAL_PT" ] && echo found || echo missing))"
echo "  output_dir: $OUTPUT_DIR"
echo "  max_k:      $MAX_K"
echo "  epochs:     $NUM_EPOCHS"
echo ""

python -m newline_experiment.train_probes \
    --train_dataset "$TRAIN_PT" \
    "${VAL_ARG[@]}" \
    --output_dir    "$OUTPUT_DIR" \
    --max_k         "$MAX_K" \
    --num_epochs    "$NUM_EPOCHS" \
    --batch_size    "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay  "$WEIGHT_DECAY" \
    --device        "$DEVICE"

echo ""
echo "✓ Results saved to $OUTPUT_DIR/experiment_results.json"
