#!/bin/bash
# Create training datasets from The Pile
# Can be run from any directory

set -e

# Get absolute path to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Add src to PYTHONPATH so package is importable
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

echo "Creating Pile datasets..."
echo "Output directory: $PROJECT_ROOT/data/"
echo ""

python -m utils.create_pile_datasets \
    --dataset_name monology/pile-uncopyrighted \
    --model_name meta-llama/Llama-3.2-1B \
    --output_dir "$PROJECT_ROOT/data" \
    --n_train 10000 \
    --n_val 2000 \
    --n_small_train 50 \
    --n_small_val 10 \
    --min_tokens 64 \
    --max_tokens 256 \
    --seed 42

echo ""
echo "✓ Done! Datasets created in $PROJECT_ROOT/data/"
