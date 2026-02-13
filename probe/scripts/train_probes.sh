#!/bin/bash
# Train probes on pre-extracted activations (Step 2 of decoupled pipeline).
# Does NOT require the language model; reads .pt files produced by build_dataset.sh.
# Can be run from any directory.
#
# Override defaults via env vars, e.g.:
#   NUM_EPOCHS=20 LEARNING_RATE=1e-3 bash probe/scripts/train_probes.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

export PYTHONPATH="$PROJECT_ROOT/probe/src:$PYTHONPATH"

TRAIN_DATASET="${TRAIN_DATASET:-$PROJECT_ROOT/probe/data/activations_train.pt}"
VAL_DATASET="${VAL_DATASET:-$PROJECT_ROOT/probe/data/activations_val.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/probe/results/experiment_results_linear}"
MAX_K=3
PROBE_TYPE=linear
NUM_EPOCHS=10
BATCH_SIZE=512
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-3
DEVICE="${DEVICE:-cuda}"

if [ ! -f "$TRAIN_DATASET" ]; then
    echo "ERROR: Training activations not found at $TRAIN_DATASET"
    echo "Run build_dataset.sh first."
    exit 1
fi

echo "Training probes from: $PROJECT_ROOT"
echo "Train dataset: $TRAIN_DATASET"
echo "Output dir:    $OUTPUT_DIR"
echo ""

TRAIN_CMD=(
    python -m look_ahead_probe.train_probes
    --train_dataset "$TRAIN_DATASET"
    --max_k "$MAX_K"
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
echo "✓ Training complete! Results in $OUTPUT_DIR/"
