"""
Model registry / factory for building models from Hydra config.
"""

from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig

from src.networks.resnet18 import LinearResNet18
from src.networks.simple_cnn import LinearSimpleCNN

_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "LinearResNet18": LinearResNet18,
    "LinearSimpleCNN": LinearSimpleCNN,
}


def build_model(cfg: DictConfig, ret_emb: bool = True) -> nn.Module:
    """
    Instantiate a model from full Hydra config.

    Uses ``cfg.model.arch`` to look up the class and passes relevant kwargs.
    """
    arch = cfg.model.arch
    cls = _MODEL_REGISTRY.get(arch)
    if cls is None:
        raise ValueError(
            f"Unknown model arch '{arch}'. Available: {list(_MODEL_REGISTRY)}"
        )

    head_bias = cfg.model.get("head", {}).get("bias", True)
    model = cls(
        emb_dim=cfg.model.embedding_dim,
        num_classes=cfg.dataset.num_classes,
        p_dropout=cfg.model.p_dropout,
        use_dropout=cfg.model.use_dropout,
        ret_emb=ret_emb,
        head_bias=head_bias,
    )
    return model


def unwrap(model: nn.Module) -> nn.Module:
    """Strip DataParallel wrapper if present."""
    return model.module if isinstance(model, nn.DataParallel) else model
