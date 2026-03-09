#!/usr/bin/env bash
# One-time conda env setup on the MATX cluster.
# Run this on a login node (sc) — it only installs packages, no GPU needed.
#
# Usage: bash steering/scripts/slurm_setup.sh
set -euo pipefail

ENV_NAME="steering"
ENV_DIR="/matx/u/$USER/conda/envs/$ENV_NAME"

# Ensure conda uses matx drive
conda config --add pkgs_dirs "/matx/u/$USER/conda/pkgs" 2>/dev/null || true
conda config --add envs_dirs "/matx/u/$USER/conda/envs" 2>/dev/null || true

if [ -d "$ENV_DIR" ]; then
    echo "Conda env '$ENV_NAME' already exists at $ENV_DIR"
    echo "To recreate: conda env remove -n $ENV_NAME && bash $0"
    exit 0
fi

echo "Creating conda env '$ENV_NAME' ..."
conda create -n "$ENV_NAME" python=3.11 -y

echo "Installing packages ..."
conda run -n "$ENV_NAME" pip install torch transformers accelerate pronouncing matplotlib numpy

echo ""
echo "Done! Activate with: conda activate $ENV_NAME"
