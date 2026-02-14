#!/bin/bash
# Plot poem probe results (Step 3 of decoupled pipeline).
# Auto-discovers all trained i-position results under RESULTS_DIR.
# Can be run from any directory.
#
# Override defaults via env vars, e.g.:
#   RESULTS_DIR=/path/to/results ACC_MAX=0.3 bash poem/scripts/plot_results.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# poem/src needed for visualize.py; probe/src for look_ahead_probe imports
export PYTHONPATH="$PROJECT_ROOT/poem/src:$PROJECT_ROOT/probe/src:$PYTHONPATH"

RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/poem/results/experiment_results_linear}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/poem/results/plots}"
ACC_MIN="${ACC_MIN:-0}"
ACC_MAX="${ACC_MAX:-0.5}"

mkdir -p "$OUTPUT_DIR"

# ------------------------------------------------------------------
# Auto-discover all i-position result JSONs, sorted numerically
# (i_neg8, i_neg7, ..., i_neg1, i0, i1, i2, ...) via Python
# ------------------------------------------------------------------
ORDERED_JSONS=$(python - "$RESULTS_DIR" <<'EOF'
import sys, re
from pathlib import Path

results_dir = Path(sys.argv[1])
jsons = list(results_dir.glob("*/experiment_results.json"))

def sort_key(p):
    name = p.parent.name          # e.g. "i_neg3", "i0", "i2"
    m = re.match(r"i_neg(\d+)$", name)
    if m:
        return -int(m.group(1))   # i_neg8 → -8
    m = re.match(r"i(\d+)$", name)
    if m:
        return int(m.group(1))    # i0 → 0, i2 → 2
    return 0

jsons.sort(key=sort_key)
for j in jsons:
    print(j)
EOF
)

if [ -z "$ORDERED_JSONS" ]; then
    echo "ERROR: No experiment_results.json found under $RESULTS_DIR"
    echo "Run train_probes.sh first."
    exit 1
fi

mapfile -t ORDERED <<< "$ORDERED_JSONS"

echo "Plotting ${#ORDERED[@]} i-position result(s) → $OUTPUT_DIR"
echo "Accuracy y-axis: [$ACC_MIN, $ACC_MAX]"
echo ""

python -m visualize \
    "${ORDERED[@]}" \
    --show-val \
    --show-top5 \
    --show-rhyme \
    --acc-min "$ACC_MIN" \
    --acc-max "$ACC_MAX" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "✓ Plots saved to $OUTPUT_DIR/"
