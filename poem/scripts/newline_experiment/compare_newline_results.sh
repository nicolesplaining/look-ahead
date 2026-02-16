#!/bin/bash
# Compare newline_experiment probe results across k values on one plot.
# Splits the single experiment_results.json into per-k slices, then
# calls compare_results.py to overlay them.
#
# Override via env vars:
#   RESULTS_DIR=/path/to/newline_experiment/dir
#   OUTPUT_DIR=/path/to/output
#   METRIC=val           (val | top5 | rhyme | rhyme5)
#   MAX_K=5
#
# Optional argument:
#   --file_name my_plot.png

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

export PYTHONPATH="$PROJECT_ROOT/poem/src:$PROJECT_ROOT/probe/src:$PYTHONPATH"

RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/poem/results/qwen3-32B/newline_experiment}"
RESULT_JSON="$RESULTS_DIR/experiment_results.json"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_DIR/plots}"
METRIC=val
MAX_K=2
FILE_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --file_name) FILE_NAME="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ ! -f "$RESULT_JSON" ]; then
    echo "ERROR: $RESULT_JSON not found. Run train_newline_probes.sh first."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ------------------------------------------------------------------
# Split the single JSON into one file per k (written to a temp dir)
# ------------------------------------------------------------------
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

python3 - <<EOF
import json, sys

with open("$RESULT_JSON") as f:
    data = json.load(f)

for k in range($MAX_K + 1):
    k_results = {key: val for key, val in data["results"].items() if val["k"] == k}
    if not k_results:
        continue
    out = {
        "config":   data.get("config", {}),
        "metadata": data.get("metadata", {}),
        "results":  k_results,
    }
    path = "$TMPDIR/k{}.json".format(k)
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"  k={k}: {len(k_results)} layer entries → {path}")
EOF

# ------------------------------------------------------------------
# Build argument lists — one entry per k
# ------------------------------------------------------------------
COLORS=(tomato steelblue seagreen)
JSONS=()
LABELS=()
USED_COLORS=()

for k in $(seq 0 "$MAX_K"); do
    f="$TMPDIR/k${k}.json"
    if [ -f "$f" ]; then
        JSONS+=("$f")
        LABELS+=("k=${k}")
        USED_COLORS+=("${COLORS[$k % ${#COLORS[@]}]}")
    fi
done

if [ ${#JSONS[@]} -eq 0 ]; then
    echo "ERROR: no per-k files found"
    exit 1
fi

NAME_ARG=()
[ -n "$FILE_NAME" ] && NAME_ARG=(--file_name "$FILE_NAME")

echo ""
echo "Comparing ${#JSONS[@]} k values, metric=$METRIC → $OUTPUT_DIR"
echo ""

python -m compare_results \
    "${JSONS[@]}" \
    --metric    "$METRIC" \
    --labels    "${LABELS[@]}" \
    --colors    "${USED_COLORS[@]}" \
    --acc-min   0 \
    --acc-max   0.5 \
    --output-dir "$OUTPUT_DIR" \
    "${NAME_ARG[@]}"

echo ""
echo "✓ Plot saved to $OUTPUT_DIR/"
