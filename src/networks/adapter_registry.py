from __future__ import annotations

import torch.nn as nn

from src.networks.heads import (
    LinearAdapter,
    TwoLayerMLPAdapter,
    ThreeLayerMLPAdapter,
    FourLayerMLPAdapter,
)

ADAPTER_REGISTRY: dict[str, type[nn.Module]] = {
    "meta_lr": LinearAdapter,
    "meta_mlp_2": TwoLayerMLPAdapter,
    "meta_mlp_3": ThreeLayerMLPAdapter,
    "meta_mlp_4": FourLayerMLPAdapter,
}


def build_adapter(
    meta_type: str, input_dim: int, num_classes: int, bias: bool = True, **kwargs
) -> nn.Module:
    """Build an adapter module based on its meta_type string."""
    cls = ADAPTER_REGISTRY.get(meta_type)
    if cls is None:
        raise ValueError(
            f"Unknown meta_type '{meta_type}'. Available: {list(ADAPTER_REGISTRY)}"
        )

    # Some heads require specific params that others don't handle well, we
    # standardise instantiation across the signature variations in `heads.py`.
    if meta_type == "meta_lr":
        return cls(emb_dim=input_dim, num_classes=num_classes, bias=bias)
    elif meta_type == "meta_mlp_2":
        return cls(
            in_dim=input_dim,
            hidden_dim=kwargs.get("hidden_dim", 128),
            num_classes=num_classes,
            dropout=kwargs.get("dropout", 0.0),
            bias=bias,
        )
    elif meta_type in ("meta_mlp_3", "meta_mlp_4"):
        return cls(in_dim=input_dim, num_classes=num_classes, bias=bias)

    raise ValueError(f"Unhandled meta_type logic for initialization: {meta_type}")


def adapter_architecture_name(meta_type: str) -> str:
    """Get the string name of the adapter class for MLflow logging."""
    cls = ADAPTER_REGISTRY.get(meta_type)
    if cls is None:
        return "UnknownAdapter"
    return cls.__name__
