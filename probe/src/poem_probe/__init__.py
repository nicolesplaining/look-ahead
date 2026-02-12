"""Probe whether models know the rhyming word after reading the first line of a couplet."""

from .extract_poem_dataset import extract_poem_activations
from .train_poem_probe import train_poem_probes

__all__ = [
    "extract_poem_activations",
    "train_poem_probes",
]

# poem_experiment.py is executable via python -m, not imported
