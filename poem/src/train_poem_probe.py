#!/usr/bin/env python3
"""
Train probes for poem rhyme prediction at a specific position i.

Reads the i-indexed .pt dataset produced by extract_poem_dataset.py.
Filters samples where i_values == train_position, then trains one probe
per layer — reusing look_ahead_probe.probe and look_ahead_probe.train_probe.

Outputs:
    probes/probe_layer_{L}_i{i}_{type}.pt   (if save_weights=True)
    training_summary_poem.pt
    experiment_results.json  (same schema as probe experiment, compatible with
                              look_ahead_probe.visualize_results; uses i as 'k')
"""

from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

from look_ahead_probe.probe import FutureTokenProbe
from look_ahead_probe.train_probe import train_probe, evaluate_probe


def train_all_layers_at_position(
    train_position: int,
    train_dataset_path: str,
    val_dataset_path: Optional[str] = None,
    probe_type: str = "linear",
    num_epochs: int = 10,
    batch_size: int = 512,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-3,
    output_dir: str = "poem_results",
    device: str = "cuda",
    save_weights: bool = False,
    tokenizer=None,
) -> dict:
    """
    Train probes on all layers using activations at position i = train_position.

    Poems that don't have a sample at train_position (e.g. short first lines
    when train_position is a large negative number) are simply not included.

    Returns all_results dict keyed by "layer{L}_i{train_position}".
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading training dataset...")
    train_data = torch.load(train_dataset_path, weights_only=False)
    train_layer_acts = train_data['layer_activations']
    train_targets = train_data['targets']
    train_i_values = train_data['i_values']
    metadata = train_data['metadata']

    val_data = None
    if val_dataset_path is not None:
        print("Loading validation dataset...")
        val_data = torch.load(val_dataset_path, weights_only=False)

    available_layers = sorted(train_layer_acts.keys())

    # Filter to the requested position
    train_mask = train_i_values == train_position
    n_train = int(train_mask.sum())
    if n_train == 0:
        raise ValueError(
            f"No training samples found for i={train_position}. "
            f"Available i range: {int(train_i_values.min())} to {int(train_i_values.max())}"
        )

    val_mask = None
    n_val = 0
    if val_data is not None:
        val_mask = val_data['i_values'] == train_position
        n_val = int(val_mask.sum())

    print(f"\nLayers: {len(available_layers)} ({available_layers[0]}–{available_layers[-1]})")
    print(f"Position: i={train_position}")
    print(f"Train samples: {n_train}  |  Val samples: {n_val}\n")

    train_targets_pos = train_targets[train_mask].to(device=device, dtype=torch.long)
    val_targets_pos = (
        val_data['targets'][val_mask].to(device=device, dtype=torch.long)
        if val_data is not None and n_val > 0 else None
    )

    all_results = {}

    for layer_idx in available_layers:
        train_acts = (train_layer_acts[layer_idx][train_mask]
                      .to(device=device, dtype=torch.float32))

        probe = FutureTokenProbe(
            input_dim=metadata['d_model'],
            vocab_size=metadata['vocab_size'],
            probe_type=probe_type,
        ).to(device)

        train_loader = DataLoader(
            TensorDataset(train_acts, train_targets_pos),
            batch_size=batch_size, shuffle=True,
        )

        val_loader = None
        if val_targets_pos is not None:
            val_acts = (val_data['layer_activations'][layer_idx][val_mask]
                        .to(device=device, dtype=torch.float32))
            val_loader = DataLoader(
                TensorDataset(val_acts, val_targets_pos),
                batch_size=batch_size,
            )

        history = train_probe(
            probe=probe,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
        )

        results = None
        decoded_predictions: Optional[List[dict]] = None
        if val_loader is not None:
            results = evaluate_probe(probe, val_loader, device=device)
            if tokenizer is not None:
                decoded_predictions = [
                    {
                        "predicted": tokenizer.decode([int(p)]).strip(),
                        "target":    tokenizer.decode([int(t)]).strip(),
                    }
                    for p, t in zip(results["predictions"], results["targets"])
                ]

        save_path = None
        if save_weights:
            save_path = str(output_dir / f"probe_layer_{layer_idx}_i{train_position}_{probe_type}.pt")
            torch.save({
                'probe_state_dict': probe.state_dict(),
                'layer_idx': layer_idx,
                'train_position': train_position,
                'probe_type': probe_type,
                'metadata': metadata,
            }, save_path)

        key = f"layer{layer_idx}_i{train_position}"
        all_results[key] = {
            'layer': layer_idx,
            'k': train_position,   # 'k' field used by visualize_results.py for grouping
            'history': history,
            'results': results,
            'decoded_predictions': decoded_predictions,
            'save_path': save_path,
        }

        train_acc = history['train_acc'][-1]
        msg = f"  Layer {layer_idx:2d}: train={train_acc:.4f}"
        if results:
            msg += f"  val={results['accuracy']:.4f}"
        print(msg)

    return all_results


def make_experiment_results_json(all_results: dict, metadata: dict, config: dict) -> dict:
    """Format all_results into experiment_results.json schema."""
    results = {}
    for key, data in all_results.items():
        entry = {
            'layer': data['layer'],
            'k': data['k'],
            'train_accuracy': float(data['history']['train_acc'][-1]),
            'train_loss': float(data['history']['train_loss'][-1]),
        }
        if data.get('results') is not None:
            entry['val_accuracy'] = float(data['results']['accuracy'])
            entry['val_top5_accuracy'] = float(data['results']['top5_accuracy'])
            entry['val_loss'] = float(data['results']['loss'])
        if data.get('decoded_predictions') is not None:
            entry['decoded_predictions'] = data['decoded_predictions']
        if data.get('save_path'):
            entry['probe_path'] = data['save_path']
        results[key] = entry

    return {
        'config': config,
        'metadata': metadata,
        'results': results,
    }
