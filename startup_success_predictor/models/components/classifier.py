"""MLP Classifier for startup success prediction."""

from typing import override

import torch.nn as nn
from torch import Tensor


class MLPClassifier(nn.Module):
    """Multi-layer perceptron for binary classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int = 1,
        leaky_relu_slope: float = 0.2,
        use_batch_norm: bool = True,
        dropout: float = 0.3,
    ) -> None:
        """Initialize MLP classifier."""
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers: list[nn.Module] = []
        in_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(leaky_relu_slope))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        # Output layer (no activation, will use BCEWithLogitsLoss)
        layers.append(nn.Linear(in_dim, output_dim))

        self.model = nn.Sequential(*layers)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through MLP."""
        out = self.model(x)
        assert isinstance(out, Tensor)
        return out
