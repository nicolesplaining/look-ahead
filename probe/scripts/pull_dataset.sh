#!/bin/bash
# Pull probe activation datasets from HuggingFace Hub.
# Requires: huggingface-cli login  (or HF_TOKEN env var set)
#
# Usage:
#   HF_REPO=username/look-ahead-activations bash probe/scripts/pull_dataset.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

HF_REPO="${HF_REPO:-nick-rui/probe-data}"

DATA_DIR="$PROJECT_ROOT/probe/data"
mkdir -p "$DATA_DIR"

echo "Pulling probe datasets from $HF_REPO ..."

python - <<EOF
import shutil, sys
from huggingface_hub import hf_hub_download

def pull(filename, dest):
    try:
        cached = hf_hub_download("${HF_REPO}", filename, repo_type="dataset")
        shutil.copy2(cached, dest)
        print(f"✓ Downloaded {filename}")
    except Exception as e:
        print(f"  (skipped {filename}: {e})")

pull("probe/activations_train.pt", "${DATA_DIR}/activations_train.pt")
pull("probe/activations_val.pt",   "${DATA_DIR}/activations_val.pt")
EOF

echo ""
echo "✓ Done. Datasets in $DATA_DIR/"
