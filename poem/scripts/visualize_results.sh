#!/bin/bash
# Visualize a single experiment result JSON — one line per metric.
# Can be run from any directory.
#
# Override via env vars:
#   RESULTS_BASE=/path/to/results/dir
#   I_VAL=0              (which i-position to visualize)
#   RESULT_JSON=/custom/path/experiment_results.json
#   OUTPUT_DIR=/path/to/output
#
# Optional argument:
#   --file_name my_plot.png

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

export PYTHONPATH="$PROJECT_ROOT/poem/src:$PROJECT_ROOT/probe/src:$PYTHONPATH"

RESULTS_BASE="${RESULTS_BASE:-$PROJECT_ROOT/poem/results/qwen3-32B}"
I_VAL="${I_VAL:-0}"
RESULT_JSON="${RESULT_JSON:-$RESULTS_BASE/i${I_VAL}/experiment_results.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_BASE/plots}"
ACC_MIN=0
ACC_MAX=1

FILE_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --file_name) FILE_NAME="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ ! -f "$RESULT_JSON" ]; then
    echo "ERROR: $RESULT_JSON not found"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

NAME_ARG=()
[ -n "$FILE_NAME" ] && NAME_ARG=(--file_name "$FILE_NAME")

echo "Visualizing: $RESULT_JSON"
echo ""

python -m visualize_results \
    "$RESULT_JSON" \
    --show-val \
    --show-top5 \
    --show-rhyme \
    --show-rhyme5 \
    --acc-min "$ACC_MIN" \
    --acc-max "$ACC_MAX" \
    --output-dir "$OUTPUT_DIR" \
    "${NAME_ARG[@]}"

echo ""
echo "✓ Plot saved to $OUTPUT_DIR/"
