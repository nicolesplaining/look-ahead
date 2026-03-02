#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$ROOT_DIR/activation-patching/initial_experiment.py"
LOG_DIR="$ROOT_DIR/activation-patching/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
NOHUP_LOG="$LOG_DIR/nohup_master_${TIMESTAMP}.log"

EXPERIMENT_IDS=(
  "exp1_newline"
  "exp1_but"
  "exp1_then"
  "exp1_he"
  "exp2_newline_asymmetric_context"
  "exp3_og_clean_newline_zero"
  "exp4_og_clean_newline_donor"
)

if [[ "${1:-}" == "--worker" ]]; then
  cd "$ROOT_DIR"
  echo "Worker started at $(date)"
  echo "Running experiments sequentially..."
  for exp_id in "${EXPERIMENT_IDS[@]}"; do
    exp_log="$LOG_DIR/${TIMESTAMP}_${exp_id}.log"
    echo
    echo "=== [$exp_id] starting at $(date) ==="
    echo "Log: $exp_log"
    python "$SCRIPT_PATH" --experiment-id "$exp_id" 2>&1 | tee "$exp_log"
    echo "=== [$exp_id] finished at $(date) ==="
  done
  echo
  echo "All experiments completed at $(date)"
  exit 0
fi

nohup bash "$0" --worker > "$NOHUP_LOG" 2>&1 &
PID=$!

echo "Started sequential activation-patching runs in background."
echo "PID: $PID"
echo "Master log: $NOHUP_LOG"
echo "Follow progress with:"
echo "  tail -f \"$NOHUP_LOG\""
