#!/usr/bin/env bash
# Run steering with L2-normalized vectors on a reduced set of pairs.
# Steers scheme 0 -> {5, 6, 7} only (3 pairs instead of 90).
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export SOURCE="0"
export TARGET="5 6 7"
export NORMALIZE=1
export ALPHA="${ALPHA:-5.0}"

exec bash "$SCRIPT_DIR/run_steering.sh" "$@"
