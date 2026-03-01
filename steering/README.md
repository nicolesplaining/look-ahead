# Steering Vector Experiments

Tests whether a steering vector constructed from a rhyme-priming prompt pair
can steer an unrelated neutral prompt toward a target rhyme family.

**Steering vector:** `v = h_clean - h_corrupt` at the newline token, where
`h_clean` comes from a prompt ending with "sleep" ("-eep" family) and
`h_corrupt` from the same prompt ending with "rest" ("-est" family).
The vector is added to the residual stream of a **neutral prompt** (no rhyme
cue) during prefill. Because of KV caching, this single intervention persists
into all decode steps.

---

## Files

| File | Purpose |
|---|---|
| `utils.py` | Shared config, model loading, CMU dict, steering vector extraction, hook, generation, evaluation |
| `layer_sweep.py` | Sweep all layers at fixed α; find which layer maximally steers toward "-eep" |
| `alpha_sweep.py` | At a fixed layer, sweep α values; find minimum effective steering strength |

---

## Usage

**Step 1 — Layer sweep** (run first; takes ~N_SAMPLES × num_layers generations)
```bash
python layer_sweep.py [--alpha 1.0] [--n_samples 200]
# Outputs: steering_results/layer_sweep.json, figures/layer_sweep.{png,pdf}
# Prints best layer at the end → use as --layer in step 2
```

**Step 2 — Alpha sweep** (run at the best layer from step 1)
```bash
python alpha_sweep.py --layer <best_layer> [--n_samples 200]
# Outputs: steering_results/alpha_sweep.json, figures/alpha_sweep.{png,pdf}
```

Override model:
```bash
python layer_sweep.py --model_name Qwen/Qwen3-8B --n_samples 50
```

---

## Key config (edit `utils.py`)

```python
CLEAN_PROMPT   = "A rhyming couplet:\nShe closed her eyes to get some sleep,\n"
CORRUPT_PROMPT = "A rhyming couplet:\nShe closed her eyes to get some rest,\n"
NEUTRAL_PROMPT = "He walked out into the open air,\n"

LAYER_SWEEP_ALPHA  = 1.0
ALPHA_SWEEP_VALUES = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
N_SAMPLES = 200
```

---

## What to check before a full run

```python
# 1. Baseline rhyme rates near zero for both families (neutral prompt is truly neutral)
# 2. Steering vector norms > 0 (printed on startup)
# 3. Hook fires assertion passes (layer_sweep.py line: assert hook_fired[0])
# Quick smoke test:
python layer_sweep.py --n_samples 5
```

---

## Output structure

```
steering_results/
    layer_sweep.json          # per-layer clean/corrupt rates + completions
    layer_sweep_partial.json  # incremental save (safe to inspect mid-run)
    alpha_sweep.json
figures/
    layer_sweep.{png,pdf}
    alpha_sweep.{png,pdf}
```

---

## Interpretation

- **Layer sweep peak** — the layer where "-eep" rate is highest shows where
  the rhyme plan is most injectable. Compare to probe accuracy peak.
- **Alpha sweep** — minimum α for significant "-eep" lift above baseline.
  Watch that "-est" rate does not also rise (would indicate nonspecific effect).
- **Convergence** — if peak steering layer ≈ peak probe layer, this is
  evidence that the same representation is both readable (probing) and
  injectable (steering).
