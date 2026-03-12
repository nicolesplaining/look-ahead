#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
PYTHONPATH=""

# ── configurable ─────────────────────────────────────────────────────────────
RESULTS_PATH="${RESULTS_PATH:-$PROJECT_ROOT/steering/results/Qwen3-32B/results.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/steering/results/Qwen3-32B}"
# Optional: comma-separated id:name pairs, e.g. "0:at,1:ight,2:ing"
SCHEME_NAMES="${SCHEME_NAMES:-}"
# Optional: figure size as WxH, e.g. "14x5"
FIGSIZE="12x2"
# Optional: plot title string
TITLE=""
# Optional: axis labels
XLABEL="${XLABEL:-}"
YLABEL="Fraction of \nSteered Rhymes"

EXTRA_ARGS=("$@")

SCHEME_NAMES_FLAG=()
if [ -n "$SCHEME_NAMES" ]; then
    SCHEME_NAMES_FLAG=(--scheme-names "$SCHEME_NAMES")
fi

FIGSIZE_FLAG=()
if [ -n "$FIGSIZE" ]; then
    FIGSIZE_FLAG=(--figsize "$FIGSIZE")
fi

TITLE_FLAG=()
if [ -n "$TITLE" ]; then
    TITLE_FLAG=(--title "$TITLE")
fi

XLABEL_FLAG=()
if [ -n "$XLABEL" ]; then
    XLABEL_FLAG=(--xlabel "$XLABEL")
fi

YLABEL_FLAG=()
if [ -n "$YLABEL" ]; then
    YLABEL_FLAG=(--ylabel "$YLABEL")
fi

export PYTHONPATH="$PROJECT_ROOT/steering/src:$PYTHONPATH"

python -m steering_probe.plot_results \
    --results-path "$RESULTS_PATH" \
    --output-dir   "$OUTPUT_DIR" \
    "${SCHEME_NAMES_FLAG[@]+"${SCHEME_NAMES_FLAG[@]}"}" \
    "${FIGSIZE_FLAG[@]+"${FIGSIZE_FLAG[@]}"}" \
    "${TITLE_FLAG[@]+"${TITLE_FLAG[@]}"}" \
    "${XLABEL_FLAG[@]+"${XLABEL_FLAG[@]}"}" \
    "${YLABEL_FLAG[@]+"${YLABEL_FLAG[@]}"}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
