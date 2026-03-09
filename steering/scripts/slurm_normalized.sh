#!/usr/bin/env bash
#SBATCH --job-name=steer-norm
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --cpus-per-gpu=10
#SBATCH --time=24:00:00
#SBATCH -w matx1
#SBATCH --output=/matx/u/%u/steering-norm-%j.log
#SBATCH --error=/matx/u/%u/steering-norm-%j.log

# Normalized steering experiment on MATX cluster (batch job).
# Submit: sbatch steering/scripts/slurm_normalized.sh
# Monitor: tail -f /matx/u/$USER/steering-norm-<jobid>.log
set -euo pipefail

echo "=== Normalized Steering Experiment (SLURM) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

export HF_HOME="/matx/u/$USER/huggingface"
mkdir -p "$HF_HOME"

# ── compute vectors if missing ────────────────────────────────────────────────
VECTORS_PATH="$PROJECT_ROOT/steering/results/steering_vectors.pt"
if [ ! -f "$VECTORS_PATH" ]; then
    echo ">>> Steering vectors not found — computing them first ..."
    echo "    Started at $(date)"
    bash steering/scripts/compute_vectors.sh
    echo "    Finished at $(date)"
else
    echo ">>> Reusing existing steering vectors: $VECTORS_PATH"
fi

# ── activate env ─────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate steering

echo "Python: $(which python)"
python -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"

# ── run ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR="$PROJECT_ROOT/steering/results/normalized"
mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="$PROJECT_ROOT/steering/src:${PYTHONPATH:-}"

# Use device_map=auto to shard across 2x L40S
python -m steering_probe.run_steering \
    --model          "Qwen/Qwen3-32B" \
    --vectors-path   "$PROJECT_ROOT/steering/results/steering_vectors.pt" \
    --data-path      "$PROJECT_ROOT/steering/data/poems-val.jsonl" \
    --output-dir     "$OUTPUT_DIR" \
    --alpha          5.0 \
    --max-new-tokens 20 \
    --gen-vector-pos 0 \
    --positions      -5 -4 -3 -2 -1 0 \
    --gen-positions  1 2 3 \
    --source         0 \
    --target         5 6 7 \
    --device         auto \
    --dtype          bfloat16 \
    --normalize

echo ""
echo ">>> Plotting results ..."
python -m steering_probe.plot_results \
    --results-path "$OUTPUT_DIR/results.json" \
    --output-dir   "$OUTPUT_DIR/plots"

echo ""
echo "=== DONE at $(date) ==="
echo "Results: $OUTPUT_DIR/"
