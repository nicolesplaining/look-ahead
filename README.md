# look-ahead

Train probes to predict future tokens from language model activations.

## Quick Start

```bash
# 1. Check model compatibility (optional)
python -m look_ahead_probe.check_model --model_name gpt2-small

# 2. Build activation datasets
python -m look_ahead_probe.build_look_ahead_activation_dataset \
    --model_name gpt2-small \
    --dataset_path data/train.jsonl \
    --max_k 10 \
    --output_path datasets/train.pt

# 3. Train probes on all layers
python -m look_ahead_probe.train_all_layers \
    --train_dataset datasets/train.pt \
    --val_dataset datasets/val.pt \
    --k 5 \
    --output_dir results/

# Or run complete experiment pipeline
python -m look_ahead_probe.layer_k_experiment \
    --model_name meta-llama/Llama-3.2-1B \
    --train_dataset_path data/train.jsonl \
    --val_dataset_path data/val.jsonl \
    --max_k 5 \
    --probe_type mlp \
    --num_epochs 10

# Or visualize results
python -m look_ahead_probe.visualize_results \
    --results_path experiment_results/experiment_results.json \
    --output_dir experiment_results/
```

## Repository Structure

```
look-ahead/
├── src/look_ahead_probe/              # Main package
│   # Library modules (imported by users)
│   ├── probe.py                       # Probe architectures (linear, MLP)
│   ├── data_loading.py                # Data utilities
│   ├── activation_extraction.py       # Activation extraction
│   ├── train_probe.py                 # Training functions
│   # Executable modules (run via python -m)
│   ├── check_model.py                 # Model compatibility checker
│   ├── build_look_ahead_activation_dataset.py  # Dataset builder
│   ├── train_all_layers.py            # Multi-layer trainer
│   ├── layer_k_experiment.py          # Full experiment pipeline
│   └── visualize_results.py           # Results visualization
├── scripts/                           # Shell script wrappers
│   ├── run_layer_k_experiment.sh      # Quick test script
│   └── check_model.sh                 # Model checker wrapper
├── data/                              # Training/validation data
│   ├── example_train.jsonl
│   └── example_val.jsonl
└── results/                           # Experiment outputs (git-ignored)
```

## Key Features

- **Multi-k extraction**: One dataset contains targets for all k values (k=1 to max_k)
- **Multi-layer extraction**: Extract all model layers in a single pass
- **Reusable datasets**: Run expensive inference once, train many probes
- **Modular design**: Library modules for programmatic use, executable modules for CLI

## Python API

```python
from look_ahead_probe import (
    generate_and_extract_all_layers,
    load_extracted_dataset,
    FutureTokenProbe,
    train_probe,
    evaluate_probe,
)
from transformer_lens import HookedTransformer

# Extract activations
model = HookedTransformer.from_pretrained("gpt2-small")
layer_acts, targets, texts = generate_and_extract_all_layers(
    model=model, prompts=["Example"], max_k=10
)

# Load and train
acts, targets, metadata = load_extracted_dataset("dataset.pt", layer_idx=6, k=5)
probe = FutureTokenProbe(metadata['d_model'], metadata['vocab_size'], "linear")
train_probe(probe, train_loader, val_loader, num_epochs=10)
```

## Documentation

- See `src/look_ahead_probe/README.md` for detailed package documentation
- See `scripts/README.md` for experiment scripts documentation

