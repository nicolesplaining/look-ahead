# look_ahead_probe

Train probes to predict future tokens from language model activations.

## Quick Start

**Note:** All commands should be run from the repository root directory.

```bash
# 0. Check model (optional but recommended)
python -m look_ahead_probe.check_model \
    --model_name gpt2-small

# 1. Build dataset (extracts all layers, targets for k=1,2,...,max_k)
python -m look_ahead_probe.build_look_ahead_activation_dataset \
    --model_name gpt2-small \
    --dataset_path data.jsonl \
    --split_value train \
    --max_k 10 \
    --output_path datasets/train.pt

# 2. Train probes on all layers for k=5
python -m look_ahead_probe.train_all_layers \
    --train_dataset datasets/train.pt \
    --val_dataset datasets/val.pt \
    --k 5 \
    --output_dir results/
```

## Key Features

- **Multi-k extraction**: One dataset contains targets for all k values (k=1 to max_k)
- **Multi-layer extraction**: Extract all model layers in a single pass
- **Reusable datasets**: Run expensive inference once, train many probes

## Files

**Library modules** (imported by users):
- `probe.py` - Probe architectures (linear, MLP)
- `data_loading.py` - Dataset loading utilities
- `activation_extraction.py` - Efficient multi-layer, multi-k extraction
- `train_probe.py` - Training and evaluation functions

**Executable modules** (callable via `python -m look_ahead_probe.<module>`):
- `check_model.py` - Inspect model and verify extraction assumptions
- `build_look_ahead_activation_dataset.py` - Extract activations and save to disk
- `train_all_layers.py` - Train probes on all layers for a k value
- `layer_k_experiment.py` - End-to-end experiment pipeline (check → extract → train)
- `visualize_results.py` - Generate plots from experiment results

## Dataset Format

JSONL with `text` field and optional `split` field:
```json
{"text": "Example text", "split": "train"}
```

## Python API

```python
from look_ahead_probe import (
    generate_and_extract_all_layers,
    load_extracted_dataset,
    FutureTokenProbe,
    train_probe,
)
from transformer_lens import HookedTransformer

# Extract activations
model = HookedTransformer.from_pretrained("gpt2-small")
layer_acts, targets, texts = generate_and_extract_all_layers(
    model=model, prompts=["Example"], max_k=10
)
# targets shape: [n_samples, 10] for k=1,2,...,10

# Load and train
acts, targets, metadata = load_extracted_dataset("dataset.pt", layer_idx=6, k=5)
probe = FutureTokenProbe(metadata['d_model'], metadata['vocab_size'], "linear")
# ... train probe
```
