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
TRAIN_POSITION=0                        # i=0 = first-line \n; negative = earlier in first line
MODEL_NAME=Qwen/Qwen3-32B            # set to enable decoded_predictions in JSON
PROBE_TYPE="${PROBE_TYPE:-linear}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
BATCH_SIZE=32
LEARNING_RATE=1e-4
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
if [ -n "$MODEL_NAME" ]; then
    TRAIN_CMD+=(--model_name "$MODEL_NAME")
fi

"${TRAIN_CMD[@]}"

# Print a results summary table from the JSON
if [ "$TRAIN_POSITION" -ge 0 ] 2>/dev/null; then
    I_LABEL="i${TRAIN_POSITION}"
else
    I_LABEL="i_neg${TRAIN_POSITION#-}"
fi
RESULTS_FILE="$OUTPUT_DIR/$I_LABEL/experiment_results.json"

if [ -f "$RESULTS_FILE" ]; then
python - <<EOF
import json

with open("$RESULTS_FILE") as f:
    data = json.load(f)

entries = sorted(data["results"].values(), key=lambda x: x["layer"])
has_val    = any("val_accuracy"        in e for e in entries)
has_rhyme  = any("rhyme_accuracy"      in e for e in entries)
has_rhyme5 = any("top5_rhyme_accuracy" in e for e in entries)

cols = [("Layer", 5), ("Train Acc", 9)]
if has_val:    cols += [("Val Acc", 8), ("Top-5", 6)]
if has_rhyme:  cols += [("Rhyme@1", 7)]
if has_rhyme5: cols += [("Rhyme@5", 7)]
header = " | ".join(f"{name:>{w}}" for name, w in cols)
sep    = "=" * len(header)

print("\n" + sep)
print(header)
print("-" * len(sep))

best_layer, best_acc = None, -1.0
for e in entries:
    row = f"{e['layer']:>5} | {e['train_accuracy']:>9.4f}"
    if has_val:
        val_str  = f"{e['val_accuracy']:.4f}"      if "val_accuracy"      in e else "      —"
        top5_str = f"{e['val_top5_accuracy']:.4f}" if "val_top5_accuracy" in e else "      —"
        row += f" | {val_str:>8} | {top5_str:>6}"
        if e.get("val_accuracy", -1.0) > best_acc:
            best_acc, best_layer = e["val_accuracy"], e["layer"]
    if has_rhyme:
        row += f" | {e['rhyme_accuracy']:.4f}" if "rhyme_accuracy" in e else " |       —"
    if has_rhyme5:
        row += f" | {e['top5_rhyme_accuracy']:.4f}" if "top5_rhyme_accuracy" in e else " |       —"
    print(row)

print(sep)
if best_layer is not None:
    print(f"Best val acc:    Layer {best_layer}  ({best_acc:.4f})")
if has_rhyme:
    best_r1 = max(e.get("rhyme_accuracy", -1.0) for e in entries)
    best_r1l = next(e["layer"] for e in entries if e.get("rhyme_accuracy") == best_r1)
    print(f"Best Rhyme@1:    Layer {best_r1l}  ({best_r1:.4f})")
if has_rhyme5:
    best_r5 = max(e.get("top5_rhyme_accuracy", -1.0) for e in entries)
    best_r5l = next(e["layer"] for e in entries if e.get("top5_rhyme_accuracy") == best_r5)
    print(f"Best Rhyme@5:    Layer {best_r5l}  ({best_r5:.4f})")
EOF
fi

echo ""
echo "✓ Poem training complete! Results in $OUTPUT_DIR/"
