"""Probe architectures for future token prediction."""

import torch.nn as nn


class FutureTokenProbe(nn.Module):
    """Probe for predicting future tokens."""

    def __init__(self, input_dim: int, vocab_size: int, probe_type: str = "linear"):
        """Initialize probe. probe_type: 'linear' or 'mlp'."""
        super().__init__()
        self.probe_type = probe_type

        if probe_type == "linear":
            # Simple linear probe: directly map activations to logits
            self.probe = nn.Linear(input_dim, vocab_size)
        elif probe_type == "mlp":
            # MLP probe: add nonlinearity and capacity
            self.probe = nn.Sequential(
                nn.Linear(input_dim, input_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(input_dim * 2, vocab_size)
            )
        else:
            raise ValueError(f"Unknown probe type: {probe_type}. Choose 'linear' or 'mlp'.")

    def forward(self, x):
        """Forward pass: x [batch_size, input_dim] -> logits [batch_size, vocab_size]."""
        return self.probe(x)
