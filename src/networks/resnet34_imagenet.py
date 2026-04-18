"""
ResNet34 backbone variants for large-image datasets (e.g. ImageNet100).

Architecture shared by both variants:
    Input (224×224)
      → ResNet34 (standard stride-2)
      → 512-dim GAP
      → Linear(512 → emb_dim)   ← topographic embedding (e.g. 16×16 grid when emb_dim=256)
      → ReLU                    ← classifier branch only; topographic loss sees pre-ReLU
      → Dropout
      → Linear(emb_dim → num_classes)

Two classes are provided, differing only in backbone initialisation:
- ``FinetuneResNet34`` — backbone loaded from ImageNet1KV1 pretrained weights.
- ``ScratchResNet34``  — backbone initialised randomly (standard Kaiming/He init).

``forward`` returns ``(embeddings, logits)`` when ``ret_emb=True``:
- ``embeddings``: pre-ReLU neck output — used by the topographic loss and profiling.
- ``logits``:     post-ReLU dropout classifier output.

Both match the ``(embeddings, logits)`` contract of ``LinearResNet18``.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torchvision.models import ResNet34_Weights


def _make_resnet34_backbone(pretrained: bool) -> nn.Module:
    """Return a ResNet34 with the final FC replaced by Identity (exposes 512-dim GAP)."""
    weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = tvm.resnet34(weights=weights)
    backbone.fc = nn.Identity()
    return backbone


class FinetuneResNet34(nn.Module):
    """ResNet34 fine-tuned from ImageNet1KV1 pretrained weights.

    Backbone is initialised from torchvision's ImageNet1KV1 checkpoint.
    Intended for fine-tuning on a target dataset (e.g. ImageNet100).

    Args:
        emb_dim:     Topographic embedding dimension (neck output).
                     Default 256 → 16×16 grid via ``get_grid_shape``.
        num_classes: Number of output classes.
        p_dropout:   Dropout probability applied before the classifier.
        use_dropout: If False, dropout is replaced with an identity (no-op).
        ret_emb:     If True, ``forward`` returns ``(embeddings, logits)``.
        head_bias:   Whether the final linear classifier has a bias term.
    """

    def __init__(
        self,
        emb_dim: int = 256,
        num_classes: int = 100,
        p_dropout: float = 0.5,
        use_dropout: bool = True,
        ret_emb: bool = False,
        head_bias: bool = True,
    ):
        super().__init__()
        self.ret_emb = ret_emb
        self.backbone = _make_resnet34_backbone(pretrained=True)
        self.neck = nn.Linear(512, emb_dim)
        self.dropout = nn.Dropout(p_dropout) if use_dropout else nn.Identity()
        self.classifier = nn.Linear(emb_dim, num_classes, bias=head_bias)

    def forward(self, x):
        gap = self.backbone(x)  # (B, 512)
        embeddings = self.neck(gap)  # (B, emb_dim) — topographic grid, pre-ReLU
        activated = F.relu(embeddings)
        logits = self.classifier(self.dropout(activated))
        return (embeddings, logits) if self.ret_emb else logits


class ScratchResNet34(nn.Module):
    """ResNet34 trained from random initialisation (no pretrained weights).

    Backbone uses the standard torchvision ResNet34 architecture with default
    Kaiming/He initialisation. Suitable for training from scratch on any dataset.

    Args:
        emb_dim:     Topographic embedding dimension (neck output).
                     Default 256 → 16×16 grid via ``get_grid_shape``.
        num_classes: Number of output classes.
        p_dropout:   Dropout probability applied before the classifier.
        use_dropout: If False, dropout is replaced with an identity (no-op).
        ret_emb:     If True, ``forward`` returns ``(embeddings, logits)``.
        head_bias:   Whether the final linear classifier has a bias term.
    """

    def __init__(
        self,
        emb_dim: int = 256,
        num_classes: int = 100,
        p_dropout: float = 0.5,
        use_dropout: bool = True,
        ret_emb: bool = False,
        head_bias: bool = True,
    ):
        super().__init__()
        self.ret_emb = ret_emb
        self.backbone = _make_resnet34_backbone(pretrained=False)
        self.neck = nn.Linear(512, emb_dim)
        self.dropout = nn.Dropout(p_dropout) if use_dropout else nn.Identity()
        self.classifier = nn.Linear(emb_dim, num_classes, bias=head_bias)

    def forward(self, x):
        gap = self.backbone(x)  # (B, 512)
        embeddings = self.neck(gap)  # (B, emb_dim) — topographic grid, pre-ReLU
        activated = F.relu(embeddings)
        logits = self.classifier(self.dropout(activated))
        return (embeddings, logits) if self.ret_emb else logits
