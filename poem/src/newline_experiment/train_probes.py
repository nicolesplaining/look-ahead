#!/usr/bin/env python3
"""
Train linear probes on i=0 (newline) activations to predict tokens at
k steps before the rhyme word (k=0 = rhyme word, k=1 = one before, ...).

Reads newline_{train,val}.pt produced by build_newline_dataset.py.
Outputs experiment_results.json compatible with compare_results.py /
visualize_results.py (uses k as the 'k' field in the JSON schema).

Usage:
    python -m newline_experiment.train_probes \\
        --train_dataset poem/data/newline_train.pt \\
        --val_dataset   poem/data/newline_val.pt \\
        --output_dir    poem/results/newline_experiment \\
        --max_k 5
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

from look_ahead_probe.probe import FutureTokenProbe
from look_ahead_probe.train_probe import train_probe, evaluate_probe


# ---------------------------------------------------------------------------
# Training loop for one k
# ---------------------------------------------------------------------------

def _train_k(
    k: int,
    train_acts: Dict[int, torch.Tensor],  # {layer: Tensor[M, d]}
    train_targets_k: torch.Tensor,        # Tensor[M_valid]
    train_valid_mask: torch.Tensor,
    val_acts: Optional[Dict[int, torch.Tensor]],
    val_targets_k: Optional[torch.Tensor],
    val_valid_mask: Optional[torch.Tensor],
    metadata: dict,
    probe_type: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: str,
    output_dir: Path,
    save_weights: bool,
) -> dict:
    available_layers = sorted(train_acts.keys())
    n_train = int(train_valid_mask.sum())
    n_val   = int(val_valid_mask.sum()) if val_valid_mask is not None else 0

    print(f"\n── k={k} ──────────────────────────────────────────")
    print(f"  Train: {n_train}  |  Val: {n_val}")

    all_results = {}

    for layer_idx in available_layers:
        tr_acts = (train_acts[layer_idx][train_valid_mask]
                   .to(device=device, dtype=torch.float32))
        tr_tgts = train_targets_k.to(device=device, dtype=torch.long)

        probe = FutureTokenProbe(
            input_dim=metadata['d_model'],
            vocab_size=metadata['vocab_size'],
            probe_type=probe_type,
        ).to(device)

        train_loader = DataLoader(
            TensorDataset(tr_acts, tr_tgts),
            batch_size=batch_size, shuffle=True,
        )

        val_loader = None
        if val_acts is not None and val_valid_mask is not None and n_val > 0:
            va_acts = (val_acts[layer_idx][val_valid_mask]
                       .to(device=device, dtype=torch.float32))
            va_tgts = val_targets_k.to(device=device, dtype=torch.long)
            val_loader = DataLoader(
                TensorDataset(va_acts, va_tgts),
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

        val_results = None
        if val_loader is not None:
            val_results = evaluate_probe(probe, val_loader, device=device)

        save_path = None
        if save_weights:
            save_path = str(output_dir / "probes" / f"probe_layer_{layer_idx}_k{k}_{probe_type}.pt")
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'probe_state_dict': probe.state_dict(),
                'layer_idx': layer_idx,
                'k': k,
                'probe_type': probe_type,
                'metadata': metadata,
            }, save_path)

        key = f"layer{layer_idx}_k{k}"
        all_results[key] = {
            'layer':   layer_idx,
            'k':       k,
            'history': history,
            'results': val_results,
            'save_path': save_path,
        }

        train_acc = history['train_acc'][-1]
        msg = f"  Layer {layer_idx:2d}: train={train_acc:.4f}"
        if val_results:
            msg += f"  val={val_results['accuracy']:.4f}  top5={val_results['top5_accuracy']:.4f}"
        print(msg)

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train newline_experiment probes (fixed i=0, varying k)"
    )
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--val_dataset",   type=str, default=None)
    parser.add_argument("--output_dir",    type=str, required=True)
    parser.add_argument("--max_k",         type=int, default=5)
    parser.add_argument("--probe_type",    type=str, default="linear")
    parser.add_argument("--num_epochs",    type=int, default=10)
    parser.add_argument("--batch_size",    type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay",  type=float, default=1e-3)
    parser.add_argument("--device",        type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_weights",  action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load datasets
    # -----------------------------------------------------------------------
    print(f"Loading train: {args.train_dataset}")
    train_data = torch.load(args.train_dataset, weights_only=False)
    train_acts    = train_data['layer_activations']  # {layer: Tensor[M, d]}
    train_targets = train_data['targets']            # Tensor[M, max_k+1]
    metadata      = train_data['metadata']
    stored_max_k  = metadata.get('max_k', train_targets.shape[1] - 1)
    max_k = min(args.max_k, stored_max_k)

    val_acts    = None
    val_targets = None
    if args.val_dataset:
        print(f"Loading val:   {args.val_dataset}")
        val_data    = torch.load(args.val_dataset, weights_only=False)
        val_acts    = val_data['layer_activations']
        val_targets = val_data['targets']

    available_layers = sorted(train_acts.keys())
    print(f"\nModel: {metadata.get('model_name', 'unknown')}")
    print(f"Layers: {available_layers[0]}–{available_layers[-1]}")
    print(f"Poems (train): {train_targets.shape[0]}")
    print(f"max_k to train: {max_k}")

    config = {
        'train_dataset': args.train_dataset,
        'val_dataset':   args.val_dataset,
        'max_k':         max_k,
        'probe_type':    args.probe_type,
        'num_epochs':    args.num_epochs,
        'batch_size':    args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay':  args.weight_decay,
    }

    # -----------------------------------------------------------------------
    # Train for each k
    # -----------------------------------------------------------------------
    all_results = {}

    for k in range(max_k + 1):
        train_valid_mask = (train_targets[:, k] != -1)
        train_targets_k  = train_targets[train_valid_mask, k]

        val_valid_mask = None
        val_targets_k  = None
        if val_targets is not None:
            val_valid_mask = (val_targets[:, k] != -1)
            val_targets_k  = val_targets[val_valid_mask, k]

        k_results = _train_k(
            k=k,
            train_acts=train_acts,
            train_targets_k=train_targets_k,
            train_valid_mask=train_valid_mask,
            val_acts=val_acts,
            val_targets_k=val_targets_k,
            val_valid_mask=val_valid_mask,
            metadata=metadata,
            probe_type=args.probe_type,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            output_dir=output_dir,
            save_weights=args.save_weights,
        )
        all_results.update(k_results)

    # -----------------------------------------------------------------------
    # Write experiment_results.json
    # -----------------------------------------------------------------------
    results_json = {}
    for key, data in all_results.items():
        entry = {
            'layer':          data['layer'],
            'k':              data['k'],
            'train_accuracy': float(data['history']['train_acc'][-1]),
            'train_loss':     float(data['history']['train_loss'][-1]),
        }
        if data.get('results') is not None:
            entry['val_accuracy']      = float(data['results']['accuracy'])
            entry['val_top5_accuracy'] = float(data['results']['top5_accuracy'])
            entry['val_loss']          = float(data['results']['loss'])
        if data.get('save_path'):
            entry['probe_path'] = data['save_path']
        results_json[key] = entry

    output = {
        'config':   config,
        'metadata': {k: v for k, v in metadata.items()
                     if k != 'layer_activations'},
        'results':  results_json,
    }

    json_path = output_dir / "experiment_results.json"
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Results saved to {json_path}")

    # Summary table
    print("\n── Summary (best val layer per k) ──────────────────")
    header = f"{'k':>4}  {'Best Layer':>10}  {'Train':>8}  {'Val':>8}  {'Top-5':>8}"
    print(header)
    for k in range(max_k + 1):
        k_entries = [
            (key, v) for key, v in results_json.items()
            if v['k'] == k and 'val_accuracy' in v
        ]
        if not k_entries:
            continue
        best_key, best = max(k_entries, key=lambda x: x[1]['val_accuracy'])
        print(
            f"  k={k:<2}  layer={best['layer']:>4}  "
            f"train={best['train_accuracy']:.4f}  "
            f"val={best['val_accuracy']:.4f}  "
            f"top5={best['val_top5_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
