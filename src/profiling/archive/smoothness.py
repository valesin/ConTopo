"""
Profiling: spatial smoothness (Moran's I) of activation maps.
"""

from __future__ import annotations

import torch
from src.losses.topographic import get_grid_shape


def morans_i(activations: torch.Tensor, emb_dim: int) -> float:
    """
    Global Moran's I for activations arranged on a 2-D grid.

    ``activations`` shape: [N, emb_dim] — average over samples first, then
    compute spatial autocorrelation of the mean activation per unit.
    """
    mean_act = activations.mean(dim=0).cpu().float()  # [emb_dim]
    h, w = get_grid_shape(emb_dim)
    grid = mean_act.reshape(h, w)

    n = h * w
    x_bar = grid.mean()
    z = grid - x_bar

    # Build adjacency (4-connected) weights
    W_sum = 0.0
    numerator = 0.0
    for i in range(h):
        for j in range(w):
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    W_sum += 1.0
                    numerator += z[i, j] * z[ni, nj]

    denom = float((z ** 2).sum())
    if denom == 0 or W_sum == 0:
        return 0.0
    return float(n / W_sum * numerator / denom)
