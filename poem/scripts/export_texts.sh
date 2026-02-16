#!/bin/bash
# Export generated_texts from activation .pt files to JSONL.
# Can be run from any directory.
#
# Override defaults via env vars, e.g.:
#   SPLIT=train bash poem/scripts/export_texts.sh
#   PT_PATH=/custom/path.pt OUTPUT=/custom/out.jsonl bash poem/scripts/export_texts.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

export PYTHONPATH="$PROJECT_ROOT/poem/src:$PROJECT_ROOT/probe/src:$PYTHONPATH"

DATA_DIR="$PROJECT_ROOT/poem/data"

# Run for train and val by default; set SPLIT=train or SPLIT=val to do one only
SPLIT="${SPLIT:-both}"

run_export() {
    local pt="$1"
    local out="$2"
    if [ ! -f "$pt" ]; then
        echo "  (skipped: $pt not found)"
        return
    fi
    python -m export_texts "$pt" --output "$out"
}

echo "Exporting generated texts from: $DATA_DIR"
echo ""

if [ -n "$PT_PATH" ]; then
    # Manual override: export a single file
    OUTPUT="${OUTPUT:-${PT_PATH%.pt}.jsonl}"
    run_export "$PT_PATH" "$OUTPUT"
elif [ "$SPLIT" = "train" ]; then
    run_export "$DATA_DIR/activations_train.pt" "$DATA_DIR/generated_train.jsonl"
elif [ "$SPLIT" = "val" ]; then
    run_export "$DATA_DIR/activations_val.pt" "$DATA_DIR/generated_val.jsonl"
else
    run_export "$DATA_DIR/activations_train.pt" "$DATA_DIR/generated_train.jsonl"
    run_export "$DATA_DIR/activations_val.pt"   "$DATA_DIR/generated_val.jsonl"
fi

echo ""
echo "✓ Done."
