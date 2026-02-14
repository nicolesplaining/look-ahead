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

The dataset stores activations for `i = -MAX_BACK ... target_i` (inclusive) for each poem.
Poems are dropped only if no second-line newline is generated. Short first lines just won't have samples at large negative i values — the training script skips missing positions automatically.

## Pipeline

---

### Step 1 — Extract activations (needs GPU + model)

```bash
bash poem/scripts/build_dataset.sh
# → poem/data/activations_train.pt
#   poem/data/activations_val.pt
```

Key env vars: `MODEL_NAME`, `MAX_BACK` (default 8), `MAX_NEW_TOKENS`, `MAX_TRAIN_PROMPTS`, `DEVICE`

The `.pt` file schema:
```
{
  layer_activations: {layer_idx → Tensor[N, d_model]},   # bfloat16
  targets:           Tensor[N],   # rhyming word token ID (same target for all positions of a poem)
  i_values:          Tensor[N],   # position index relative to first-line \n (can be negative)
  generated_texts:   List[str],   # one entry per kept poem
  metadata:          {model_name, max_back, n_poems, n_samples, d_model, vocab_size, layers, i_range, ...}
}
```
`N` = total (position, poem) pairs; each poem contributes multiple samples (one per valid position i).

To push/pull from HuggingFace:
```bash
bash poem/scripts/push_dataset.sh   # upload .pt files to nick-rui/probe-data
bash poem/scripts/pull_dataset.sh   # download them
```

---

### Step 2 — Train probes at a specific position (no model needed)

```bash
# Train at i=0 (the first-line \n token)
bash poem/scripts/train_probes.sh

# Train at i=-3 (3 tokens before the \n)
TRAIN_POSITION=-3 bash poem/scripts/train_probes.sh

# Train at i=2 (2 tokens into the second line)
TRAIN_POSITION=2 bash poem/scripts/train_probes.sh
```

Each run creates its own subdirectory (`i0/`, `i_neg3/`, `i2/`, ...) inside `OUTPUT_DIR` with an `experiment_results.json`.

Key env vars: `TRAIN_DATASET`, `VAL_DATASET`, `OUTPUT_DIR`, `TRAIN_POSITION` (required), `PROBE_TYPE`, `NUM_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `WEIGHT_DECAY`

---

### Step 3 — Plot and compare positions

```bash
# Auto-discovers all trained positions and overlays them
bash poem/scripts/plot_results.sh
# → poem/results/plots/*.png
# Shows val, top-5, and rhyme accuracy on a single plot per i-position.

# Manual overlay with custom labels/colors
python -m visualize \
    poem/results/experiment_results_linear/i_neg3/experiment_results.json \
    poem/results/experiment_results_linear/i0/experiment_results.json \
    poem/results/experiment_results_linear/i2/experiment_results.json \
    --labels "i=-3" "i=0" "i=2" \
    --show-val --show-top5 --show-rhyme \
    --acc-min 0 --acc-max 0.5
```

Key env vars: `RESULTS_DIR`, `OUTPUT_DIR`, `ACC_MIN`, `ACC_MAX`

Flags: `--show-val`, `--show-top5`, `--show-rhyme` (any combination)

---

## Source layout

```
poem/src/
├── extract_poem_dataset.py   # CLI for step 1 (i-indexed multi-position extraction)
├── train_poem_probe.py       # library: train_all_layers_at_position()
├── train_probes.py           # CLI for step 2 (single i value, all layers)
└── visualize.py              # CLI for step 3 (overlay multiple i-positions)

poem/scripts/
├── build_dataset.sh          # step 1 wrapper  (env: MAX_BACK)
├── push_dataset.sh           # upload .pt to HuggingFace
├── pull_dataset.sh           # download .pt from HuggingFace
├── train_probes.sh           # step 2 wrapper  (env: TRAIN_POSITION)
└── plot_results.sh           # step 3 wrapper
```

Reuses from `look_ahead_probe`: `FutureTokenProbe`, `train_probe`, `evaluate_probe`, `visualize_results`.
