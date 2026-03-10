"""
Adapter / meta-learner heads for ensemble experiments.
"""

import torch
import torch.nn as nn



class LinearAdapter(nn.Module):
    """Linear adapter head for meta-learner regression (optionally without bias)."""

    def __init__(self, emb_dim: int = 256, num_classes: int = 10, bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_classes, bias=bias)

    def forward(self, x):
        return self.fc(x)



class ThreeLayerMLPAdapter(nn.Module):
    """Three-layer MLP adapter for meta-learning on ensemble features (optionally without bias in final layer)."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_classes: int = 10, dropout: float = 0.3, bias: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes, bias=bias),
        )

    def forward(self, x):
        return self.net(x)
