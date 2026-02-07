#!/bin/bash
# Quick test of layer-k experiment pipeline
# Can be run from any directory

set -e

# Get absolute path to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Validate required files exist
if [ ! -f "$PROJECT_ROOT/data/example_train.jsonl" ]; then
    echo "ERROR: Training data not found at $PROJECT_ROOT/data/example_train.jsonl"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/data/example_val.jsonl" ]; then
    echo "ERROR: Validation data not found at $PROJECT_ROOT/data/example_val.jsonl"
    exit 1
fi

echo "Running layer-k experiment from: $PROJECT_ROOT"
echo "Running layer-k experiment (test mode with small parameters)..."
echo ""

# Add src to PYTHONPATH so package is importable
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

python -m look_ahead_probe.layer_k_experiment \
    --model_name meta-llama/Llama-3.2-1B \
    --train_dataset_path "$PROJECT_ROOT/data/example_train.jsonl" \
    --val_dataset_path "$PROJECT_ROOT/data/example_val.jsonl" \
    --max_k 3 \
    --max_new_tokens 64 \
    --probe_type mlp \
    --num_epochs 10 \
    --batch_size 128 \
    --output_dir "$PROJECT_ROOT/experiment_results_mlp"

echo ""
echo "✓ Pipeline test complete! Check $PROJECT_ROOT/experiment_results_mlp/ for outputs"
