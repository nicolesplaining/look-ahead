#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# ── configurable ─────────────────────────────────────────────────────────────
RESULTS_PATH="${RESULTS_PATH:-$PROJECT_ROOT/steering/results/results.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/steering/results/plots}"
# Optional: comma-separated id:name pairs, e.g. "0:at,1:ight,2:ing"
SCHEME_NAMES="${SCHEME_NAMES:-}"

EXTRA_ARGS=("$@")

SCHEME_NAMES_FLAG=()
if [ -n "$SCHEME_NAMES" ]; then
    SCHEME_NAMES_FLAG=(--scheme-names "$SCHEME_NAMES")
fi

export PYTHONPATH="$PROJECT_ROOT/steering/src:${PYTHONPATH:-}"

python -m steering_probe.plot_results \
    --results-path "$RESULTS_PATH" \
    --output-dir   "$OUTPUT_DIR" \
    "${SCHEME_NAMES_FLAG[@]}" \
    "${EXTRA_ARGS[@]}"
