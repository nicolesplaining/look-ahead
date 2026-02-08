#!/bin/bash
# Check model compatibility for activation extraction
# Can be run from any directory

set -e

# Get absolute path to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Add src to PYTHONPATH so package is importable
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# MODEL_NAME=meta-llama/Llama-3.2-1B
# MODEL_NAME=meta-llama/Llama-3.2-3B
MODEL_NAME=meta-llama/Llama-3.1-8B

python -m look_ahead_probe.check_model \
    --model_name $MODEL_NAME \
    --max_new_tokens 10
