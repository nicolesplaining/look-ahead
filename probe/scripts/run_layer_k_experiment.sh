#!/bin/bash
# Quick test of layer-k experiment pipeline
# Can be run from any directory

set -e

# Get absolute path to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Validate required files exist
if [ ! -f "$PROJECT_ROOT/probe/data/example_train.jsonl" ]; then
    echo "ERROR: Training data not found at $PROJECT_ROOT/probe/data/example_train.jsonl"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/probe/data/example_val.jsonl" ]; then
    echo "ERROR: Validation data not found at $PROJECT_ROOT/probe/data/example_val.jsonl"
    exit 1
fi

echo "Running layer-k experiment from: $PROJECT_ROOT"
echo "Running layer-k experiment (test mode with small parameters)..."
echo ""

# Add src to PYTHONPATH so package is importable
export PYTHONPATH="$PROJECT_ROOT/probe/src:$PYTHONPATH"

MODEL_NAME=meta-llama/Llama-3.1-8B

python -m look_ahead_probe.layer_k_experiment \
    --model_name $MODEL_NAME \
    --train_dataset_path "$PROJECT_ROOT/probe/data/train-pile.jsonl" \
    --val_dataset_path "$PROJECT_ROOT/probe/data/val-pile.jsonl" \
    --max_k 3 \
    --max_train_prompts 100 \
    --max_val_prompts 20 \
    --max_new_tokens 32 \
    --probe_type linear \
    --num_epochs 3 \
    --learning_rate 5e-4 \
    --batch_size 256 \
    --output_dir "$PROJECT_ROOT/probe/experiment_results_linear"

echo ""
echo "✓ Pipeline test complete! Check $PROJECT_ROOT/probe/experiment_results_linear/ for outputs"
