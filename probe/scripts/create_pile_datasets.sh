#!/bin/bash
# Create training datasets from The Pile
# Can be run from any directory

set -e

# Get absolute path to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Add src to PYTHONPATH so package is importable
export PYTHONPATH="$PROJECT_ROOT/probe/src:$PYTHONPATH"

echo "Creating Pile datasets..."
echo "Output directory: $PROJECT_ROOT/probe/data/"
echo ""

python -m utils.create_pile_datasets \
    --dataset_name monology/pile-uncopyrighted \
    --model_name Qwen/Qwen3-1.7B \
    --output_dir "$PROJECT_ROOT/probe/data" \
    --subsets "Wikipedia_(en),OpenWebText2,Gutenberg_(PG-19),PubMed_Abstracts,HackerNews,PhilPapers" \
    --n_train 1000 \
    --n_val 200 \
    --n_small_train 50 \
    --n_small_val 10 \
    --min_tokens 32 \
    --max_tokens 64 \
    --seed 42

echo ""
echo "✓ Done! Datasets created in $PROJECT_ROOT/probe/data/"
