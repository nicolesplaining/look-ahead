#!/bin/bash
# Pull poem activation datasets from HuggingFace Hub.
# Requires: huggingface-cli login  (or HF_TOKEN env var set)
#
# Usage:
#   MODEL_NAME=Qwen/Qwen2.5-7B bash poem/scripts/pull_dataset.sh
#   HF_REPO=username/repo MODEL_NAME=Qwen/Qwen2.5-7B bash poem/scripts/pull_dataset.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

HF_REPO="${HF_REPO:-nick-rui/probe-data}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-32B}"

# Replace '/' with '--' for use in the repo path (e.g. Qwen/Qwen2.5-7B → Qwen--Qwen2.5-7B)
MODEL_SLUG="${MODEL_NAME//\//"--"}"
REMOTE_DIR="poem/${MODEL_SLUG}"

DATA_DIR="$PROJECT_ROOT/poem/data"
mkdir -p "$DATA_DIR"

echo "Pulling poem datasets from $HF_REPO/$REMOTE_DIR ..."

python - <<EOF
import shutil
from huggingface_hub import hf_hub_download

def pull(filename, dest):
    try:
        cached = hf_hub_download("${HF_REPO}", filename, repo_type="dataset")
        shutil.copy2(cached, dest)
        print(f"✓ Downloaded {filename}")
    except Exception as e:
        print(f"  (skipped {filename}: {e})")

pull("${REMOTE_DIR}/activations_train.pt", "${DATA_DIR}/activations_train.pt")
pull("${REMOTE_DIR}/activations_val.pt",   "${DATA_DIR}/activations_val.pt")
EOF

echo ""
echo "✓ Done. Datasets in $DATA_DIR/"
