# Poem rhyme probe

Probe whether LM residual stream activations encode the rhyming word of a couplet, at various positions relative to the end of the first line.

## Position indexing

Poem prompts have the format:
```
A rhyming couplet:
{First Line}
```
`i = 0` is the `\n` token at the end of the first line (last prompt token).
`i < 0` are tokens earlier in the first line.
`i > 0` are generated tokens in the second line.
**Target** = last token before the terminating `\n` of the second line (the rhyming word).

The dataset stores activations for `i = -MAX_BACK ... target_i` (inclusive) for each poem. Poems are dropped only if no second-line newline is generated.

---

## Experiments

### 1. Main probe experiment

Train probes across all layers at each i-position, predicting the rhyme word token.

**Step 1 — Extract activations** (GPU + model)
```bash
bash poem/scripts/build_dataset.sh
# → poem/data/activations_train.pt
#   poem/data/activations_val.pt
```
Key env vars: `MODEL_NAME`, `MAX_BACK` (default 8), `MAX_NEW_TOKENS`, `MAX_TRAIN_PROMPTS`

The `.pt` schema:
```
{
  layer_activations: {layer_idx → Tensor[N, d_model]},   # bfloat16
  targets:           Tensor[N],    # rhyming word token ID
  i_values:          Tensor[N],    # position index relative to first-line \n
  generated_texts:   List[str],
  metadata:          {model_name, max_back, n_poems, n_samples, d_model, vocab_size, layers, i_range}
}
```
`N` = total (position, poem) pairs; each poem contributes one sample per valid i.

**Step 2 — Train probes** (no model needed)
```bash
bash poem/scripts/train_probes.sh              # i=0 by default
TRAIN_POSITION=-3 bash poem/scripts/train_probes.sh
TRAIN_POSITION=2  bash poem/scripts/train_probes.sh
```
Each run writes `experiment_results.json` to its own subdirectory (`i0/`, `i_neg3/`, `i2/`, …) inside `OUTPUT_DIR`.
Key env vars: `TRAIN_DATASET`, `VAL_DATASET`, `OUTPUT_DIR`, `TRAIN_POSITION`, `PROBE_TYPE`, `NUM_EPOCHS`, `BATCH_SIZE`

**Step 3 — Visualize**
```bash
bash poem/scripts/visualize_results.sh        # single result JSON, all metrics
bash poem/scripts/compare_results.sh          # overlay multiple i-positions on one metric
```

To push/pull datasets from HuggingFace:
```bash
bash poem/scripts/push_dataset.sh
bash poem/scripts/pull_dataset.sh
```

---

### 2. Newline experiment

Fixed activation at i=0 (the first-line `\n` token), varying prediction target: k tokens before the rhyme word (k=0 = rhyme word, k=1 = one before, …, k=max_k).

Reuses the existing `.pt` files from experiment 1 — no re-extraction needed.

**Step 1 — Build per-k dataset**
```bash
bash poem/scripts/newline_experiment/build_newline_dataset.sh
# → poem/data/newline_train.pt, newline_val.pt
```

**Step 2 — Train probes per k**
```bash
bash poem/scripts/newline_experiment/train_newline_probes.sh
# → poem/results/.../newline_experiment/experiment_results.json
```

**Step 3 — Compare k values**
```bash
bash poem/scripts/newline_experiment/compare_newline_results.sh
```

---

### 3. Ablation — direct rhyme rate

Measures how often the model produces a rhyming second line (no probes). Uses the `pronouncing` library (CMU dict) for rhyme checking.

Two extraction modes:
- `with_newline`: generation terminates at `\n`; extract last word before it.
- `without_newline`: suppress `\n` tokens via `bad_words_ids`; find first punctuation mark as line terminator.

```bash
bash poem/scripts/ablation/run_ablation.sh
# → poem/results/ablation/{model}/results_with_newline.json
#                                  results_without_newline.json
# Prints a combined accuracy table (rhymed / total) at the end.
```
Key env vars: `MODEL_NAME`, `MODE` (`both` / `with_newline` / `without_newline`), `MAX_NEW_TOKENS`, `MAX_POEMS`, `OUTPUT_DIR`

---

## Source layout

```
poem/src/
├── extract_poem_dataset.py          # CLI: extract i-indexed activations (step 1)
├── train_poem_probe.py              # library: train_all_layers_at_position()
├── train_probes.py                  # CLI: train probes at one i-position (step 2)
├── visualize_results.py             # CLI: plot one result JSON, multiple metrics
├── compare_results.py               # CLI: compare multiple result JSONs on one metric
├── export_texts.py                  # export generated poem texts to JSONL
├── newline_experiment/
│   ├── build_newline_dataset.py     # filter i=0 samples; build targets for k=0..max_k
│   └── train_probes.py              # train probes per k, output experiment_results.json
└── ablation/
    └── evaluate_rhyming.py          # evaluate model rhyme rate; both modes, one model load

poem/scripts/
├── build_dataset.sh                 # step 1 (i-indexed extraction)
├── train_probes.sh                  # step 2 (single i, all layers)
├── visualize_results.sh             # step 3: visualize one result JSON
├── compare_results.sh               # step 3: overlay multiple i-positions
├── export_texts.sh
├── push_dataset.sh / pull_dataset.sh
├── newline_experiment/
│   ├── build_newline_dataset.sh
│   ├── train_newline_probes.sh
│   └── compare_newline_results.sh
└── ablation/
    └── run_ablation.sh              # runs both modes; prints combined accuracy table
```

Reuses from `look_ahead_probe`: `FutureTokenProbe`, `train_probe`, `evaluate_probe`.
