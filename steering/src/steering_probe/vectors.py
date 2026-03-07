"""
Compute, save, and load pairwise steering vectors.

Steering vector (src -> tgt) at (layer, rel_pos) =
    mean(tgt activations at that position) - mean(src activations at that position)

Adding alpha * vector to the residual stream nudges the model toward tgt's distribution.
"""
import torch
from typing import Dict, List, Optional, Tuple


SchemeMeans = Dict[int, Dict[int, Dict[int, torch.Tensor]]]
# {scheme: {layer: {rel_pos: tensor(hidden_dim)}}}

SteeringVectors = Dict[Tuple[int, int], Dict[int, Dict[int, torch.Tensor]]]
# {(src, tgt): {layer: {rel_pos: tensor(hidden_dim)}}}


def compute_steering_vectors(scheme_means: SchemeMeans) -> SteeringVectors:
    """Compute all pairwise mean-diff vectors."""
    schemes = sorted(scheme_means.keys())
    vectors: SteeringVectors = {}

    for src in schemes:
        for tgt in schemes:
            if src == tgt:
                continue
            vectors[(src, tgt)] = {}
            for l in scheme_means[src]:
                if l not in scheme_means[tgt]:
                    continue
                vectors[(src, tgt)][l] = {}
                common = set(scheme_means[src][l]) & set(scheme_means[tgt][l])
                for rel_pos in common:
                    vectors[(src, tgt)][l][rel_pos] = (
                        scheme_means[tgt][l][rel_pos] - scheme_means[src][l][rel_pos]
                    )
    return vectors


def save_vectors(
    vectors: SteeringVectors,
    scheme_means: SchemeMeans,
    path: str,
    metadata: Optional[dict] = None,
) -> None:
    torch.save(
        {"vectors": vectors, "scheme_means": scheme_means, "metadata": metadata or {}},
        path,
    )


def load_vectors(path: str) -> Tuple[SteeringVectors, SchemeMeans, dict]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["vectors"], data["scheme_means"], data["metadata"]
