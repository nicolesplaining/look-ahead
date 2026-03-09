#!/usr/bin/env bash
# One-time venv setup on the MATX cluster.
# Run this on a login node (sc).
#
# Usage: bash steering/scripts/slurm_setup.sh
set -euo pipefail

ENV_DIR="/matx/u/$USER/steering-env"

if [ -d "$ENV_DIR" ]; then
    echo "Venv already exists at $ENV_DIR"
    echo "To recreate: rm -rf $ENV_DIR && bash $0"
    exit 0
fi

echo "Creating venv at $ENV_DIR ..."
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

echo "Installing packages ..."
pip install --upgrade pip
pip install torch transformers accelerate pronouncing matplotlib numpy

echo ""
echo "Done! Activate with: source $ENV_DIR/bin/activate"
