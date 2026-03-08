# steering

Experiments on **activation steering for rhyme control** in autoregressive LMs.

A steering vector is computed as the mean-difference in residual-stream activations
between two rhyme-scheme groups.  The vector is then injected at a specific
`(layer, position i)` during generation to measure how strongly it shifts
the model's rhyme output toward the target scheme.

Position convention: `i = 0` is the final newline token of the first couplet line,
`i < 0` are earlier prompt tokens, and `i > 0` are generated tokens.

---

## Directory structure

```
steering/
├── data/
│   ├── poems-train.jsonl   ← 100 examples × 10 schemes (used to compute vectors)
│   └── poems-val.jsonl     ← 20 examples × 10 schemes (used to evaluate steering)
│
├── results/
│   ├── steering_vectors.pt ← output of step 1
│   ├── results.json        ← output of step 2
│   └── plots/              ← output of step 3 (PNG heatmaps)
│
├── scripts/
│   ├── compute_vectors.sh  ← step 1: extract activations + compute mean-diff vectors
│   ├── run_steering.sh     ← step 2: apply vectors, generate, evaluate rhyme
│   └── plot_results.sh     ← step 3: produce layer × position heatmaps
│
└── src/steering_probe/
    ├── extract.py          ← forward-pass hooks; computes per-scheme mean activations
    ├── vectors.py          ← pairwise mean-diff computation, .pt save/load
    ├── steer.py            ← hook-based steering during model.generate()
    ├── evaluate.py         ← CMU Pronouncing Dict rhyme-key evaluation
    ├── plot.py             ← matplotlib heatmap utilities
    ├── compute_vectors.py  ← CLI entry point for step 1
    ├── run_steering.py     ← CLI entry point for step 2
    └── plot_results.py     ← CLI entry point for step 3
```

---

## Data format

Both JSONL files use the same schema:

```json
{"id": 1, "scheme": 0, "text": "A rhyming couplet:\nThe cat sat on the mat,\n"}
```

- `scheme`: integer 0–9 identifying the rhyme family
- `text`: the first line of the couplet (prompt fed to the model)

---

## Quickstart

```bash
# Step 1 — requires GPU; reads poems-train.jsonl, writes steering_vectors.pt
MODEL=Qwen/Qwen3-32B bash steering/scripts/compute_vectors.sh

# Step 2 — reads poems-val.jsonl + steering_vectors.pt, writes results.json
bash steering/scripts/run_steering.sh

# Step 3 — reads results.json, writes plots/ directory
SCHEME_NAMES="0:at,1:ight,2:ing,3:old,4:and,5:ee,6:ow,7:ound,8:ake,9:ell" \
    bash steering/scripts/plot_results.sh
```

All scripts are configured via environment variables (see inside each script for the
full list) and accept additional CLI args that are forwarded to the Python entry point.

Common overrides:

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `Qwen/Qwen3-32B` | HuggingFace model name or local path |
| `CONTEXT_WINDOW` | `20` | Tokens before newline to extract (step 1) |
| `LAYERS` | *(all)* | Space-separated layer indices, e.g. `"0 16 32 48"` |
| `POSITIONS` | *(all prompt)* | Prompt positions to steer at, e.g. `"-5 -3 -1 0"` |
| `GEN_POSITIONS` | *(none)* | Generation positions to steer at, e.g. `"1 2 3"` |
| `ALPHA` | `20.0` | Steering coefficient |
| `SOURCE` / `TARGET` | *(all)* | Scheme IDs to include, e.g. `"0 1"` / `"2 3"` |
| `SCHEME_NAMES` | *(numeric IDs)* | `"0:at,1:ight,..."` for plot labels |

---

## How it works

**Step 1 — compute_vectors**

For each training example the model runs a forward pass with hooks registered on
every transformer layer.  Residual-stream activations are captured at positions
`i = -(context_window) … 0` relative to the newline token.  After processing all
examples, per-scheme means are computed and the pairwise mean-difference vectors
are saved (one vector per `(src_scheme, tgt_scheme, layer, position_i)`).

**Step 2 — run_steering**

For each validation example and each `(src→tgt, layer, i)` combination:
- A forward hook injects `alpha × vector` into the residual stream at `(layer, i)`
  during `model.generate()`.
- The generated second line is evaluated with the CMU Pronouncing Dictionary:
  the last word's rhyme key (last stressed vowel + trailing phonemes) is compared
  against the target scheme's consensus rhyme key.
- Outputs `steered_rhyme_pct` and `baseline_rhyme_pct` for every combination.

For generation positions (`i > 0`) the vector computed at `--gen-vector-pos`
(default `i = 0`, the newline token) is used, since generation-time activations
are not available during training-set extraction.

**Step 3 — plot_results**

One PNG heatmap per `(src, tgt)` pair.  X-axis = position `i`, Y-axis = layer,
color = `steered_rhyme_pct`.  Baseline accuracy is shown in the title.
