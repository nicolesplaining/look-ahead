# Poem Rhyme Probe

Test whether language models "know" the rhyming word after reading just the first line of a couplet.

## The Question

When a model finishes reading the first line of a rhyming couplet, do its internal activations already encode what the rhyme will be?

```
Input (first line):  "He saw a carrot and had to grab it,"
Model generates:     " so he reached down to try and nab it."
Question:           Did the model "know" it would rhyme with "nab it"?
```

## Approach

1. **Extract activations**: Get activation at the last token of the first line
2. **Generate completion**: Let the model generate the second line
3. **Train probe**: Linear/MLP probe from first-line activation → rhyming word
4. **Measure**: How accurately can we predict the rhyme just from first-line activations?

## Dataset Structure

- **Input**: Activation at position of last prompt token (end of first line)
- **Target**: Token ID of the last word in the second line (the rhyme)

Unlike the look-ahead probe which predicts tokens at specific future positions (k tokens ahead), this probe predicts a semantically meaningful target: the rhyming word that completes the couplet.

## Usage

### 1. Extract Dataset

```bash
# Extract from poems
python -m poem_probe.extract_poem_dataset \
    --model_name meta-llama/Llama-3.2-1B \
    --poems_path data/poems.jsonl \
    --output_path poem_data/train.pt \
    --max_new_tokens 30
```

### 2. Train Probes

```bash
# Train on all layers
python -m poem_probe.train_poem_probe \
    --train_dataset poem_data/train.pt \
    --val_dataset poem_data/val.pt \
    --probe_type mlp \
    --num_epochs 20 \
    --output_dir poem_probes/
```

## Poem Format

Poems should be in JSONL format with incomplete first lines:

```jsonl
{"id": 0, "text": "A rhyming couplet:\nHe saw a carrot and had to grab it,"}
{"id": 1, "text": "A rhyming couplet:\nThe child was afraid of the dark at night,"}
```

The model will generate the second line, and we extract:
- Activation from the comma after "grab it"
- Target: the last token of the generated line (e.g., "nab it")

## Code Reuse

This package reuses components from `look_ahead_probe`:
- `FutureTokenProbe`: Probe architectures (linear, MLP)
- `train_probe()`: Training loop
- `evaluate_probe()`: Evaluation metrics

Only the dataset extraction is custom, since we're predicting across variable-length generations rather than fixed k-ahead positions.

## Expected Results

If the model "knows" the rhyme early:
- **High accuracy** at deeper layers (near output)
- **Emerging accuracy** at middle layers
- **Low accuracy** at early layers

If the model only "figures out" the rhyme during generation:
- **Low accuracy** across all layers of the first-line activation

This tests whether rhyme knowledge is represented proactively vs. emergently during generation.

## Files

- `extract_poem_dataset.py` - Extract activations and generate completions
- `train_poem_probe.py` - Train probes on poem rhyme task
- `__init__.py` - Package exports
