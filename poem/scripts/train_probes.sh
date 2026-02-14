#!/bin/bash
# Train poem probes on pre-extracted activations (Step 2 of decoupled pipeline).
# Does NOT require the language model; reads .pt files from build_dataset.sh.
# Can be run from any directory.
#
# Override defaults via env vars, e.g.:
#   NUM_EPOCHS=30 PROBE_TYPE=mlp bash poem/scripts/train_probes.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# poem_probe imports look_ahead_probe, so both src dirs are needed
export PYTHONPATH="$PROJECT_ROOT/poem/src:$PROJECT_ROOT/probe/src:$PYTHONPATH"

TRAIN_DATASET="${TRAIN_DATASET:-$PROJECT_ROOT/poem/data/activations_train.pt}"
VAL_DATASET="${VAL_DATASET:-$PROJECT_ROOT/poem/data/activations_val.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/poem/results/experiment_results_linear}"
TRAIN_POSITION="${TRAIN_POSITION:-0}"   # i=0 = first-line \n; negative = earlier in first line
PROBE_TYPE="${PROBE_TYPE:-linear}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-3}"
DEVICE="${DEVICE:-cuda}"

if [ ! -f "$TRAIN_DATASET" ]; then
    echo "ERROR: Training activations not found at $TRAIN_DATASET"
    echo "Run build_dataset.sh first."
    exit 1
fi

echo "Training poem probes from: $PROJECT_ROOT"
echo "Train dataset:  $TRAIN_DATASET"
echo "Train position: i=$TRAIN_POSITION"
echo "Output dir:     $OUTPUT_DIR"
echo ""

TRAIN_CMD=(
    python -m train_probes
    --train_dataset "$TRAIN_DATASET"
    --train_position "$TRAIN_POSITION"
    --output_dir "$OUTPUT_DIR"
    --probe_type "$PROBE_TYPE"
    --num_epochs "$NUM_EPOCHS"
    --batch_size "$BATCH_SIZE"
    --learning_rate "$LEARNING_RATE"
    --weight_decay "$WEIGHT_DECAY"
    --device "$DEVICE"
)

if [ -f "$VAL_DATASET" ]; then
    TRAIN_CMD+=(--val_dataset "$VAL_DATASET")
fi

"${TRAIN_CMD[@]}"

echo ""
echo "✓ Poem training complete! Results in $OUTPUT_DIR/"
