# Scripts

Helper scripts for experiments and testing.

## Available Scripts

## Python Modules

The Python modules have been moved to `src/look_ahead_probe/`. They can be run using `python -m`:

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
