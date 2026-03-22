#!/bin/bash
# Plot poem probe results (Step 3 of decoupled pipeline).
# Can be run from any directory.
#
# Defaults: i=-2,-1 in yellow, i=0 in tomato, i=1..9 in blue, Rhyme@5 only.
# Override via env vars:
#   RESULTS_BASE=/path/to/results/dir
#   OUTPUT_DIR=/path/to/output
#   COLOR_I0=tomato   COLOR_REST=steelblue
#   STYLE_I0="solid"  STYLE_REST="dashed"   (any matplotlib named linestyle)
#
# Optional argument:
#   --file_name rhyme5_qwen3.png

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

export PYTHONPATH="$PROJECT_ROOT/poem/src:$PROJECT_ROOT/probe/src:$PYTHONPATH"

# MODEL_NAME=Gemma-3-27B
MODEL_NAME=Qwen3-32B
# MODEL_NAME=Llama-3.1-70B-Instruct

RESULTS_BASE="${RESULTS_BASE:-$PROJECT_ROOT/poem/results/$MODEL_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/poem/results/$MODEL_NAME/plots}"
ACC_MIN=0
ACC_MAX=1

COLOR_I0="${COLOR_I0:-tomato}"
COLOR_REST="${COLOR_REST:-steelblue}"
STYLE_I0="${STYLE_I0:-solid}"      # linestyle for i=0 (default: solid)
STYLE_REST="${STYLE_REST:-dashed}" # linestyle for i=1..9 (default: dashed)
OUTPUT_NAME=summary

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --file_name) OUTPUT_NAME="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

# ------------------------------------------------------------------
# Build argument lists
# ------------------------------------------------------------------
JSONS=()
LABELS=()
COLORS=()
STYLES=()

# i=-2,-1 — yellowish
f="$RESULTS_BASE/i_neg2/experiment_results.json"
if [ -f "$f" ]; then JSONS+=("$f"); LABELS+=("i=-2"); COLORS+=("#F4D03F"); STYLES+=("$STYLE_I0"); fi

f="$RESULTS_BASE/i_neg1/experiment_results.json"
if [ -f "$f" ]; then JSONS+=("$f"); LABELS+=("i=-1"); COLORS+=("#F4D03F"); STYLES+=("$STYLE_I0"); fi

# i=0 — distinct color and style
f="$RESULTS_BASE/i0/experiment_results.json"
if [ -f "$f" ]; then JSONS+=("$f"); LABELS+=("i=0"); COLORS+=("$COLOR_I0"); STYLES+=("$STYLE_I0"); fi

# i=1..5 — subtle light-to-dark blue gradient
# BLUE_GRADIENT=("#93C4E0" "#6AAFD4" "#4195C3" "#2676AD" "#145A96")
BLUE_GRADIENT=("#93C4E0" "#6AAFD4" "#4195C3")
for idx in 1 2 3; do
    f="$RESULTS_BASE/i${idx}/experiment_results.json"
    color="${BLUE_GRADIENT[$((idx-1))]}"
    if [ -f "$f" ]; then JSONS+=("$f"); LABELS+=("i=${idx}"); COLORS+=("$color"); STYLES+=("$STYLE_REST"); fi
done

# i=6..9 — same color and style
for idx in 6 7 8 9; do
    f="$RESULTS_BASE/i${idx}/experiment_results.json"
    if [ -f "$f" ]; then JSONS+=("$f"); LABELS+=("i=${idx}"); COLORS+=("$COLOR_REST"); STYLES+=("$STYLE_REST"); fi
done

if [ ${#JSONS[@]} -eq 0 ]; then
    echo "ERROR: No result JSONs found under $RESULTS_BASE"
    exit 1
fi

echo "Plotting ${#JSONS[@]} result(s) → $OUTPUT_DIR"
echo "Accuracy y-axis: [$ACC_MIN, $ACC_MAX]"
echo ""

COMMON_ARGS=(
    "${JSONS[@]}"
    --labels "${LABELS[@]}"
    --colors "${COLORS[@]}"
    --linestyles "${STYLES[@]}"
    --acc-min "$ACC_MIN"
    --acc-max "$ACC_MAX"
    --output-dir "$OUTPUT_DIR"
)

python -m visualize_results "${COMMON_ARGS[@]}" --show-val    --file_name "${OUTPUT_NAME}-val"
python -m visualize_results "${COMMON_ARGS[@]}" --show-top5   --file_name "${OUTPUT_NAME}-top5"
python -m visualize_results "${COMMON_ARGS[@]}" --show-rhyme  --file_name "${OUTPUT_NAME}-rhyme"
python -m visualize_results "${COMMON_ARGS[@]}" --show-rhyme5 --file_name "${OUTPUT_NAME}-rhyme5"

echo ""
echo "✓ Plots saved to $OUTPUT_DIR/"
