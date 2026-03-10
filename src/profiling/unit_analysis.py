"""
Unit-level analysis: weight norms and unit distance correlation on
a topographic grid.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.losses.topographic import get_grid_shape


def weight_norms(linear: nn.Linear) -> torch.Tensor:
    """L2 norm of each output unit's weight vector → [out_features]."""
    return linear.weight.data.norm(dim=1).cpu()


def unit_distance_correlation(linear: nn.Linear) -> torch.Tensor:
    """
    Correlation between grid distance and weight cosine similarity
    for units arranged on a 2-D grid.

    Returns: [n_pairs, 2] tensor of (distance, cosine_sim) pairs — caller
    can compute Pearson r as needed.
    """
    W = linear.weight.data.cpu().float()
    n, d = W.shape
    h, w = get_grid_shape(n)

    # grid positions
    pos = torch.stack(
        torch.meshgrid(
            torch.linspace(0, 1, h),
            torch.linspace(0, 1, w),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)
    dist = torch.cdist(pos, pos, p=2)

    # cosine similarity
    Wn = W / W.norm(dim=1, keepdim=True).clamp_min(1e-8)
    cos_sim = Wn @ Wn.t()

    # upper triangle
    idx = torch.triu_indices(n, n, offset=1)
    return torch.stack([dist[idx[0], idx[1]], cos_sim[idx[0], idx[1]]], dim=1)
