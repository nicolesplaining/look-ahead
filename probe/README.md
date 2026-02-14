# Probe

Train linear/MLP probes on LM residual stream activations to predict tokens k steps ahead, across all layers.

## Pipeline

The pipeline has three decoupled steps. Steps 1 and 2 can run on different machines.

### Step 0 — Create input data (one-time)

```bash
bash probe/scripts/create_pile_datasets.sh
# → probe/data/train-pile.jsonl, val-pile.jsonl
```

Or bring your own JSONL with a `"text"` field.

---

### Step 1 — Extract activations (needs GPU + model)

```bash
bash probe/scripts/build_dataset.sh
# → probe/data/activations_train.pt      (activations + targets, ~60 GB for 32B)
#   probe/data/activations_train.tokens.jsonl  (raw token IDs, lightweight)
#   probe/data/activations_train.texts.jsonl   (decoded text, for inspection)
# same files for val
```

Key env vars: `MODEL_NAME`, `MAX_K`, `MAX_NEW_TOKENS`, `MAX_TRAIN_PROMPTS`, `DEVICE`

The `.pt` file schema:
```
{
  layer_activations: {layer_idx → Tensor[N, d_model]},   # bfloat16
  targets:           Tensor[N, max_k],                   # token IDs
  generated_texts:   List[str],
  metadata:          {model_name, max_k, d_model, vocab_size, layers, ...}
}
```
`N` = total (position, prompt) pairs where position `i` has `max_k` valid look-ahead tokens.

To push/pull from HuggingFace:
```bash
bash probe/scripts/push_dataset.sh   # upload .pt + sidecars to nick-rui/probe-data
bash probe/scripts/pull_dataset.sh   # download them
```

---

### Step 2 — Train probes (no model needed, small GPU ok)

```bash
bash probe/scripts/train_probes.sh
# → probe/results/experiment_results_linear/experiment_results.json
```

Key env vars: `TRAIN_DATASET`, `VAL_DATASET`, `OUTPUT_DIR`, `MAX_K`, `PROBE_TYPE` (`linear`/`mlp`), `NUM_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `WEIGHT_DECAY`

Probe weights are **not** saved by default (add `--save_weights` to save them; ~9 GB per layer for 32B).

---

### Step 3 — Plot results

```bash
bash probe/scripts/plot_results.sh
# → probe/results/plots/*.png
# Overlays probe + all three baseline JSONs on a single plot per k value.
# Any of the four result files may be absent — script skips missing ones.
```

Key env vars: `PROBE_RESULTS`, `UNIGRAM_RESULTS`, `BIGRAM_RESULTS`, `TRIGRAM_RESULTS`, `OUTPUT_DIR`, `ACC_MIN`, `ACC_MAX`

---

### Baselines

N-gram (unigram/bigram/trigram) baselines using skip-k context — same "view" as the probe.

```bash
bash probe/scripts/run_baselines.sh
# → probe/results/baselines/{unigram,bigram,trigram}_results.json
# Feed any of these into plot_results.sh via RESULTS_PATH=...
```

Reads `.tokens.jsonl` by default (exact token IDs, no re-tokenization). Falls back to `.texts.jsonl` or `.pt` if needed (set `MODEL_NAME` for those).

---

## Source layout

```
probe/src/look_ahead_probe/
├── activation_extraction.py          # generate + extract residual stream
├── build_look_ahead_activation_dataset.py  # CLI for step 1
├── train_probe.py                    # single-probe training loop
├── train_all_layers.py               # train across all layers for one k
├── train_probes.py                   # CLI for step 2 (loops over k)
├── probe.py                          # FutureTokenProbe (linear / MLP)
├── data_loading.py                   # dataset loading utilities
├── baseline.py                       # n-gram baselines
├── visualize_results.py              # CLI for step 3 (multi-JSON overlay)
└── check_model.py                    # inspect model config
```
