#!/bin/bash
# Quick test of poem rhyme probe pipeline
# Can be run from any directory

set -e

# Get absolute path to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Validate required files exist
if [ ! -f "$PROJECT_ROOT/probe/data/poems.jsonl" ]; then
    echo "ERROR: Poems file not found at $PROJECT_ROOT/probe/data/poems.jsonl"
    exit 1
fi

echo "Running poem rhyme probe experiment from: $PROJECT_ROOT"
echo "Running poem experiment (100 poems, MLP probe, 20 epochs)..."
echo ""

# Add src to PYTHONPATH so package is importable
export PYTHONPATH="$PROJECT_ROOT/probe/src:$PYTHONPATH"

python -m poem_probe.poem_experiment \
    --model_name meta-llama/Llama-3.1-8B \
    --poems_path "$PROJECT_ROOT/probe/data/poems.jsonl" \
    --max_new_tokens 30 \
    --probe_type linear \
    --num_epochs 20 \
    --batch_size 16 \
    --output_dir "$PROJECT_ROOT/probe/poem_results"

echo ""
echo "✓ Poem experiment complete! Check $PROJECT_ROOT/probe/poem_results/ for outputs"
