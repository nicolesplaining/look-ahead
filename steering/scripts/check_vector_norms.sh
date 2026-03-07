#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

VECTORS_PATH="${VECTORS_PATH:-$PROJECT_ROOT/steering/results/steering_vectors.pt}"
PAIR="${PAIR:-0,5}"   # src,tgt  e.g. PAIR="2,7"
STRIDE="${STRIDE:-8}" # print every Nth layer

export PYTHONPATH="$PROJECT_ROOT/steering/src:${PYTHONPATH:-}"

python3 - <<EOF
import torch

vectors_path = "$VECTORS_PATH"
src, tgt = map(int, "$PAIR".split(","))
stride = int("$STRIDE")

v, _, meta = torch.load(vectors_path, weights_only=False)
pair = (src, tgt)

if pair not in v:
    print(f"Pair {pair} not found. Available: {list(v.keys())[:10]}")
    raise SystemExit(1)

positions = sorted(v[pair][next(iter(v[pair]))].keys())
print(f"Pair {pair}  |  positions in vectors: {positions}")
print(f"{'layer':>6}  " + "  ".join(f"||v@{p}||" for p in positions))
print("-" * (8 + 14 * len(positions)))

for layer in sorted(v[pair].keys())[::stride]:
    norms = {p: v[pair][layer][p].norm().item() for p in positions if p in v[pair][layer]}
    row = f"{layer:>6}  " + "  ".join(f"{norms.get(p, float('nan')):>8.1f}" for p in positions)
    print(row)

# summary ratios relative to pos -1
if -1 in positions:
    print()
    print("mean ||v@p|| / ||v@-1|| across layers:")
    import statistics
    for p in positions:
        if p == -1:
            continue
        ratios = [
            v[pair][l][-1].norm().item() / v[pair][l][p].norm().item()
            for l in v[pair]
            if p in v[pair][l] and v[pair][l][p].norm().item() > 0
        ]
        print(f"  -1 / {p:+d}  =  {statistics.mean(ratios):.2f}x")
EOF
