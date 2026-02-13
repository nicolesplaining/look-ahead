#!/bin/bash
# Push probe activation datasets to HuggingFace Hub.
# Requires: huggingface-cli login  (or HF_TOKEN env var set)
#
# Usage:
#   HF_REPO=username/look-ahead-activations bash probe/scripts/push_dataset.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

HF_REPO="nick-rui/probe-data"

TRAIN_PT="$PROJECT_ROOT/probe/data/activations_train.pt"
VAL_PT="$PROJECT_ROOT/probe/data/activations_val.pt"

if [ ! -f "$TRAIN_PT" ]; then
    echo "ERROR: $TRAIN_PT not found. Run build_dataset.sh first."
    exit 1
fi

echo "Pushing probe datasets to $HF_REPO ..."

python - <<EOF
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("${HF_REPO}", repo_type="dataset", exist_ok=True, private=True)

api.upload_file(
    path_or_fileobj="${TRAIN_PT}",
    path_in_repo="probe/activations_train.pt",
    repo_id="${HF_REPO}",
    repo_type="dataset",
)
print("✓ Uploaded probe/activations_train.pt")
EOF

if [ -f "$VAL_PT" ]; then
python - <<EOF
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="${VAL_PT}",
    path_in_repo="probe/activations_val.pt",
    repo_id="${HF_REPO}",
    repo_type="dataset",
)
print("✓ Uploaded probe/activations_val.pt")
EOF
fi

echo ""
echo "✓ Done. Pull with: HF_REPO=$HF_REPO bash probe/scripts/pull_dataset.sh"
