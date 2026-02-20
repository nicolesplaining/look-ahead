#!/bin/bash
# Compare one metric across i=0..9 on a single plot.
# Can be run from any directory.
#
# Override via env vars:
#   RESULTS_BASE=/path/to/results/dir
#   OUTPUT_DIR=/path/to/output
#   METRIC=rhyme5          (val | top5 | rhyme | rhyme5)
#   COLOR_I0=tomato        COLOR_REST=steelblue
#   STYLE_I0=solid         STYLE_REST=dashed   (solid|dashed|dotted|dashdot)
#
# Optional argument:
#   --file_name my_plot.png

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

export PYTHONPATH="$PROJECT_ROOT/poem/src:$PROJECT_ROOT/probe/src:$PYTHONPATH"

RESULTS_BASE=$PROJECT_ROOT/poem/results/qwen3-32B-all
OUTPUT_DIR=$PROJECT_ROOT/poem/results/qwen3-32B-all/plots
METRIC=val
ACC_MIN=0
ACC_MAX=1

COLOR_I0="${COLOR_I0:-tomato}"
COLOR_REST="${COLOR_REST:-steelblue}"
STYLE_I0="${STYLE_I0:-solid}"
STYLE_REST="${STYLE_REST:-dashed}"

FILE_NAME=summary-$METRIC

while [[ $# -gt 0 ]]; do
    case "$1" in
        --file_name) FILE_NAME="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

JSONS=()
LABELS=()
COLORS=()
STYLES=()

# i=0 — distinct color and style
f="$RESULTS_BASE/i0/experiment_results.json"
if [ -f "$f" ]; then JSONS+=("$f"); LABELS+=("i=0"); COLORS+=("$COLOR_I0"); STYLES+=("$STYLE_I0"); fi

# i=1..9 — same color and style
for idx in 1 2 3 4 5; do
    f="$RESULTS_BASE/i${idx}/experiment_results.json"
    if [ -f "$f" ]; then JSONS+=("$f"); LABELS+=("i=${idx}"); COLORS+=("$COLOR_REST"); STYLES+=("$STYLE_REST"); fi
done

if [ ${#JSONS[@]} -eq 0 ]; then
    echo "ERROR: No result JSONs found under $RESULTS_BASE"
    exit 1
fi

NAME_ARG=()
[ -n "$FILE_NAME" ] && NAME_ARG=(--file_name "$FILE_NAME")

echo "Comparing ${#JSONS[@]} result(s), metric=$METRIC → $OUTPUT_DIR"
echo ""

python -m compare_results \
    "${JSONS[@]}" \
    --metric "$METRIC" \
    --labels "${LABELS[@]}" \
    --colors "${COLORS[@]}" \
    --linestyles "${STYLES[@]}" \
    --acc-min "$ACC_MIN" \
    --acc-max "$ACC_MAX" \
    --output-dir "$OUTPUT_DIR" \
    "${NAME_ARG[@]}"

echo ""
echo "✓ Plot saved to $OUTPUT_DIR/"
