"""
Adapter / meta-learner heads for ensemble experiments.
"""

import torch
import torch.nn as nn


class LinearAdapter(nn.Module):
    """Linear adapter head for meta-learner regression (optionally without bias)."""

    def __init__(
        self,
        emb_dim: int = 256,
        num_classes: int = 10,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_classes, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TwoLayerMLPAdapter(nn.Module):
    """Two-layer MLP adapter for meta-learning on ensemble features (optionally without bias in final layer)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 10,
        dropout: float = 0.3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ThreeLayerMLPAdapter(nn.Module):
    """Three-layer MLP adapter: Input -> 128 -> ReLU -> 64 -> ReLU -> num_classes.

    Use this when a 3-layer MLP is desired. For a two-layer variant use
    `TwoLayerMLPAdapter`.
    """

    def __init__(self, in_dim: int, num_classes: int, bias: bool = True) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128, bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64, bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FourLayerMLPAdapter(nn.Module):
    """Four-layer MLP adapter: Input -> 256 -> ReLU -> 128 -> ReLU -> 64 -> ReLU -> num_classes.

    Deeper variant of ThreeLayerMLPAdapter with a wider first hidden layer.
    """

    def __init__(self, in_dim: int, num_classes: int, bias: bool = True) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256, bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128, bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64, bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
