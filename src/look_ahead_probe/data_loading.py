"""Data loading utilities."""

import json
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset


def load_jsonl_prompts(
    path: str,
    *,
    text_field: str = "text",
    split_field: Optional[str] = "split",
    split_value: Optional[str] = None,
    default_split: str = "train",
    max_examples: Optional[int] = None,
) -> List[str]:
    """Load prompts from JSONL file with optional split filtering."""
    prompts: List[str] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if p.suffix.lower() != ".jsonl":
        raise ValueError(f"Expected a .jsonl dataset file, got: {path}")

    with p.open("r", encoding="utf-8") as f:
        for line_idx, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_idx} of {path}: {e}") from e

            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object on line {line_idx} of {path}, got {type(obj).__name__}")

            if split_field is not None:
                row_split = obj.get(split_field, default_split)
                if split_value is not None and row_split != split_value:
                    continue

            text = obj.get(text_field, None)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Missing/empty '{text_field}' string on line {line_idx} of {path}. "
                    f"Got: {repr(text)}"
                )
            prompts.append(text)

            if max_examples is not None and len(prompts) >= max_examples:
                break

    return prompts


class ActivationDataset(Dataset):
    """Dataset of activations and future token targets."""

    def __init__(self, activations: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            activations: [n_samples, d_model] - activations at current position
            targets: [n_samples] - token IDs that appear k steps in the future
        """
        assert len(activations) == len(targets), "Activations and targets must have same length"
        self.activations = activations
        self.targets = targets

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return self.activations[idx], self.targets[idx]


def load_extracted_dataset(dataset_path: str, layer_idx: Optional[int] = None, k: Optional[int] = None):
    """
    Load extracted activation dataset.

    Args:
        dataset_path: Path to .pt file
        layer_idx: Layer index to load (None = all layers)
        k: Lookahead distance to use (None = return all targets)

    Returns:
        (activations, targets, metadata) where:
        - activations: [n_samples, d_model] if layer_idx specified, else dict
        - targets: [n_samples] if k specified, else [n_samples, max_k]
        - metadata: dict
    """
    data = torch.load(dataset_path)

    layer_activations = data['layer_activations']
    all_targets = data['targets']  # [n_samples, max_k]
    metadata = data['metadata']

    # Select targets for specific k if requested
    if k is not None:
        if k < 1 or k > metadata['max_k']:
            raise ValueError(f"k={k} out of range [1, {metadata['max_k']}]")
        targets = all_targets[:, k - 1]  # k=1 is index 0
    else:
        targets = all_targets

    # Select specific layer if requested
    if layer_idx is not None:
        if layer_idx not in layer_activations:
            available_layers = sorted(layer_activations.keys())
            raise ValueError(
                f"Layer {layer_idx} not in dataset. Available: {available_layers}"
            )
        return layer_activations[layer_idx], targets, metadata
    else:
        return layer_activations, targets, metadata
