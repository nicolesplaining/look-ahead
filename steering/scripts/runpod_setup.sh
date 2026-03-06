#!/usr/bin/env bash
# One-shot setup & run script for a fresh RunPod H100 instance.
#
# Usage (after cloning + checking out steering branch):
#   bash steering/scripts/runpod_setup.sh
#
# Safe to disconnect SSH after "Launching pipeline under nohup" message.
# Monitor with: tail -f steering/results/run.log
set -euo pipefail

echo "=== RunPod Setup for Steering Experiment ==="
echo "Started at $(date)"

# ── Navigate to repo root ─────────────────────────────────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

echo "Working directory: $(pwd)"

# ── Use /workspace for HF cache (root volume is tiny on RunPod) ──────────────
export HF_HOME=/workspace/tmp/hf_cache
mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME"

# ── Verify GPU ────────────────────────────────────────────────────────────────
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "ERROR: no GPU found"; exit 1; }

# ── Verify torch+CUDA (pre-installed on RunPod images) ───────────────────────
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'torch {torch.__version__}, CUDA {torch.version.cuda}, device: {torch.cuda.get_device_name(0)}')"

# ── Install remaining Python deps (don't touch torch — RunPod has it) ────────
pip install --upgrade pip
pip install transformers accelerate pronouncing matplotlib numpy
echo "Dependencies installed."

# ── Verify all imports work ───────────────────────────────────────────────────
python -c "import torch, transformers, accelerate, pronouncing, matplotlib, numpy; print('All imports OK')"

# ── Run the full pipeline (survives SSH disconnect) ───────────────────────────
LOG="$PROJECT_ROOT/steering/results/run.log"
mkdir -p "$PROJECT_ROOT/steering/results"
echo ""
echo "Launching pipeline under nohup — log at: $LOG"
echo "You can safely disconnect SSH now."
echo "Monitor with: tail -f $LOG"
echo ""
nohup bash steering/scripts/run_all.sh > "$LOG" 2>&1 &
PIPELINE_PID=$!
echo "Pipeline PID: $PIPELINE_PID"
echo "Check if still running: kill -0 $PIPELINE_PID 2>/dev/null && echo running || echo done"
