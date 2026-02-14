#!/bin/bash
# Plot poem probe results (Step 3 of decoupled pipeline).
# Can be run from any directory.
#
# Configure up to 4 result slots via env vars:
#   RESULT1=/path/to/i0/experiment_results.json
#   LABEL1="i=0"
#   COLOR1="steelblue"
#
# Example — overlay three positions:
#   RESULT1=poem/results/.../i_neg3/experiment_results.json LABEL1="i=-3" \
#   RESULT2=poem/results/.../i0/experiment_results.json    LABEL2="i=0"  \
#   RESULT3=poem/results/.../i2/experiment_results.json    LABEL3="i=2"  \
#   bash poem/scripts/plot_results.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

export PYTHONPATH="$PROJECT_ROOT/poem/src:$PROJECT_ROOT/probe/src:$PYTHONPATH"

RESULTS_BASE="${RESULTS_BASE:-$PROJECT_ROOT/poem/results/experiment_results_linear}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/poem/results/plots}"
ACC_MIN="${ACC_MIN:-0}"
ACC_MAX="${ACC_MAX:-0.5}"

# ------------------------------------------------------------------
# Result slots — set path, label, and color for each
# ------------------------------------------------------------------
RESULT1="${RESULT1:-$RESULTS_BASE/i0/experiment_results.json}"
RESULT2="${RESULT2:-}"
RESULT3="${RESULT3:-}"
RESULT4="${RESULT4:-}"

LABEL1="${LABEL1:-i=0}"
LABEL2="${LABEL2:-}"
LABEL3="${LABEL3:-}"
LABEL4="${LABEL4:-}"

COLOR1="${COLOR1:-steelblue}"
COLOR2="${COLOR2:-tomato}"
COLOR3="${COLOR3:-orange}"
COLOR4="${COLOR4:-seagreen}"

mkdir -p "$OUTPUT_DIR"

# ------------------------------------------------------------------
# Build argument lists — only include slots where the file exists
# ------------------------------------------------------------------
JSONS=()
LABELS=()
COLORS=()

if [ -f "$RESULT1" ]; then JSONS+=("$RESULT1"); LABELS+=("$LABEL1"); COLORS+=("$COLOR1"); fi
if [ -f "$RESULT2" ]; then JSONS+=("$RESULT2"); LABELS+=("$LABEL2"); COLORS+=("$COLOR2"); fi
if [ -f "$RESULT3" ]; then JSONS+=("$RESULT3"); LABELS+=("$LABEL3"); COLORS+=("$COLOR3"); fi
if [ -f "$RESULT4" ]; then JSONS+=("$RESULT4"); LABELS+=("$LABEL4"); COLORS+=("$COLOR4"); fi

if [ ${#JSONS[@]} -eq 0 ]; then
    echo "ERROR: No result JSONs found. Set RESULT1 (and optionally RESULT2–4), or run train_probes.sh first."
    exit 1
fi

echo "Plotting ${#JSONS[@]} result(s) → $OUTPUT_DIR"
echo "Accuracy y-axis: [$ACC_MIN, $ACC_MAX]"
echo ""

<<<<<<< Updated upstream
python -m visualize_results \
    "${ORDERED[@]}" \
=======
python -m visualize \
    "${JSONS[@]}" \
    --labels "${LABELS[@]}" \
    --colors "${COLORS[@]}" \
>>>>>>> Stashed changes
    --show-val \
    --show-top5 \
    --show-rhyme \
    --acc-min "$ACC_MIN" \
    --acc-max "$ACC_MAX" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "✓ Plots saved to $OUTPUT_DIR/"
