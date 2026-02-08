# Scripts

Helper scripts for experiments and testing.

## Available Scripts

## Python Modules

The Python modules have been moved to `src/look_ahead_probe/`. They can be run using `python -m`:

---

### `create_pile_datasets`
**Purpose:** Sample training data from The Pile corpus

**What it does:**
Creates training and validation datasets by sampling random token sequences from The Pile (a large, diverse text corpus). Generates both full and small datasets for testing.

**Usage:**
```bash
# Using defaults (recommended)
python -m utils.create_pile_datasets

# Custom configuration
python -m utils.create_pile_datasets \
    --dataset_name monology/pile-uncopyrighted \
    --model_name meta-llama/Llama-3.2-3B \
    --output_dir data/ \
    --n_train 10000 \
    --n_val 2000 \
    --n_small_train 50 \
    --n_small_val 10 \
    --min_tokens 64 \
    --max_tokens 256

# Sample from specific Pile subset
python -m utils.create_pile_datasets \
    --subset "Wikipedia_(en)" \
    --n_train 5000
```

**Output:**
Creates 4 JSONL files in output directory:
- `train-pile.jsonl` (10,000 sequences)
- `val-pile.jsonl` (2,000 sequences)
- `small-train-pile.jsonl` (50 sequences for quick testing)
- `small-val-pile.jsonl` (10 sequences for quick testing)

**Key arguments:**
- `--dataset_name` - HuggingFace Pile dataset (default: monology/pile-uncopyrighted)
- `--model_name` - Model for tokenizer (default: meta-llama/Llama-3.2-1B)
- `--n_train/--n_val` - Number of sequences to sample
- `--min_tokens/--max_tokens` - Sequence length range (default: 64-256)
- `--subset` - Specific Pile subset (e.g., "Wikipedia_(en)", "ArXiv")

**About The Pile:**
The Pile is an 825GB diverse text corpus containing 22 data sources including web text, scientific papers, code, books, and more. Using `monology/pile-uncopyrighted` excludes potentially copyrighted content (Books3, etc.) while retaining 95% of the data.

---

### `layer_k_experiment` ⭐
**Purpose:** End-to-end layer-k probing experiment pipeline

**What it does:**
Orchestrates the 3-step pipeline:
1. **Check model** - Verify compatibility (`check_model.py`)
2. **Build datasets** - Extract train (and optional val) activations
3. **Train & evaluate** - For all layers and k values (`train_all_layers.py`)
   - Automatically evaluates on validation set if provided
   - Saves results with train/val metrics to JSON

**Usage:**
```bash
# With validation set
python -m look_ahead_probe.layer_k_experiment \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --train_dataset_path data/train.jsonl \
    --val_dataset_path data/val.jsonl \
    --max_k 5 \
    --probe_type mlp \
    --num_epochs 10

# Training only (no validation)
python -m look_ahead_probe.layer_k_experiment \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --train_dataset_path data/example_dataset.jsonl \
    --max_k 3 \
    --skip_check
```

**JSONL format** (no split field needed):
```jsonl
{"text": "Your prompt text here"}
{"text": "Another prompt"}
```

**Output:**
```
experiment_results/
├── activations.pt              # Training activations
├── val_activations.pt          # Validation activations (if provided)
├── probes/                     # Trained probes
│   ├── k1/
│   ├── k2/
│   └── k3/
└── experiment_results.json     # Train & val results ⭐
```

**Key arguments:**
- `--model_name` - Model to probe
- `--train_dataset_path` - Training data (JSONL)
- `--val_dataset_path` - Validation data (JSONL, optional)
- `--max_k` - Maximum lookahead distance
- `--probe_type` - "linear" or "mlp"
- `--skip_check` - Skip model compatibility check

---

### `visualize_results`
**Purpose:** Visualize experimental results from layer-k probe experiments

**What it does:**
Creates separate plots for each k value showing validation accuracy across layers

**Usage:**
```bash
# Generate plots
python -m look_ahead_probe.visualize_results \
    --results_path experiment_results/experiment_results.json \
    --output_dir experiment_results/

# Include training accuracy
python -m look_ahead_probe.visualize_results \
    --results_path experiment_results/experiment_results.json \
    --output_dir experiment_results/ \
    --show_train
```

**Output:**
- `val_accuracy_k{k}.png` - Validation accuracy plots
- `train_val_accuracy_k{k}.png` - Combined train/val plots (if --show_train)

---

## Shell Script Wrappers

The shell scripts below are convenient wrappers for common tasks. All scripts use absolute paths and can be run from any directory.

### `create_pile_datasets.sh`
**Purpose:** Quick dataset creation from The Pile with default settings

**What it does:**
Calls `create_pile_datasets.py` with sensible defaults to create train/val datasets from The Pile.

**Usage:**
```bash
bash scripts/create_pile_datasets.sh
```

**Output:**
Creates 4 files in `data/`:
- `train-pile.jsonl` (10K sequences)
- `val-pile.jsonl` (2K sequences)
- `small-train-pile.jsonl` (50 sequences)
- `small-val-pile.jsonl` (10 sequences)

**Defaults:**
- Dataset: `monology/pile-uncopyrighted`
- Tokenizer: `meta-llama/Llama-3.2-1B`
- Sequence length: 64-256 tokens
- Random seed: 42

---


### `run_layer_k_experiment.sh`
**Purpose:** Quick pipeline test with small parameters

**What it does:**
Runs `layer_k_experiment.py` with small test parameters to verify the pipeline works

**Usage:**
```bash
# Can be run from any directory
bash scripts/run_layer_k_experiment.sh

# Or from anywhere
bash /path/to/look-ahead/scripts/run_layer_k_experiment.sh
```

**Test parameters:**
- Example train/val datasets
- k=1,2,3, 64 tokens
- MLP probe, 10 epochs
- Fast execution (~5-10 minutes)

---

### `run_poem_experiment.sh`
**Purpose:** End-to-end poem rhyme probe experiment

**What it does:**
Complete pipeline from raw poems to trained probes:
1. Split poems into train/val (80/20)
2. Extract activation datasets (train + val)
3. Train probes on all layers
4. Save results with accuracy per layer

**Usage:**
```bash
# Default: Llama-3.2-1B, MLP, 20 epochs
bash scripts/run_poem_experiment.sh
```

**Test parameters:**
- 100 poems (80 train, 20 val)
- MLP probe, 20 epochs
- 30 max tokens per generation
- Execution time: ~10-15 minutes (with GPU)

**Output:**
```
poem_results/
├── data/
│   ├── poems_train.jsonl   # Train split
│   ├── poems_val.jsonl     # Val split
│   ├── train.pt            # Training activations
│   └── val.pt              # Validation activations
└── probes/
    ├── layer0_probe.pt     # Probes for each layer
    ├── ...
    └── training_summary.pt # Results ⭐
```

**Expected output:** Accuracy by layer showing if model "knows" rhyme

---

### `check_model.sh`
**Purpose:** Verify model compatibility for activation extraction

**Usage:**
```bash
./check_model.sh
```

## Running Scripts

All scripts use absolute paths and can be run from any directory:

```bash
# From project root
bash scripts/run_layer_k_experiment.sh

# From anywhere
bash /path/to/look-ahead/scripts/run_layer_k_experiment.sh

# Or cd to scripts directory
cd scripts && bash run_layer_k_experiment.sh
```
