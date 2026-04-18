"""
Simple CNN backbone and end-to-end CE wrapper.
Quick to train, intended for testing pipelines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Simple CNN backbone for CIFAR-10."""

    def __init__(self, in_channels: int = 3, emb_dim: int = 256) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))

        out = self.avgpool(out)
        out = out.flatten(1)
        return self.fc(out)


class LinearSimpleCNN(nn.Module):
    """SimpleCNN encoder + dropout + linear classifier. Returns (embeddings, logits) when `ret_emb=True`."""

    def __init__(
        self,
        emb_dim: int = 256,
        num_classes: int = 10,
        p_dropout: float = 0.5,
        use_dropout: bool = True,
        ret_emb: bool = False,
        head_bias: bool = True,
    ) -> None:
        super().__init__()
        self.ret_emb = ret_emb
        self.encoder = SimpleCNN(emb_dim=emb_dim)
        self.dropout = nn.Dropout(p_dropout) if use_dropout else nn.Identity()
        self.fc = nn.Linear(emb_dim, num_classes, bias=head_bias)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encoder(x)
        logits = self.fc(self.dropout(embeddings))
        return (embeddings, logits) if self.ret_emb else logits
