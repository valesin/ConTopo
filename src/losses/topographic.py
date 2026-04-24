"""
Topographic loss functions.

Ported from ``losses/topographic.py`` — no functional changes.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ───────────────── grid helpers ─────────────────


def get_grid_shape(n_units: int) -> tuple[int, int]:
    """Return (h, w) with h*w == n_units and h as close to w as possible."""
    h = int(math.sqrt(n_units))
    while n_units % h != 0:
        h -= 1
    return h, n_units // h


def pos_dist(embedding_dim: int) -> torch.Tensor:
    """Pairwise Euclidean distance matrix for a 2-D grid of ``embedding_dim`` cells."""
    h, w = get_grid_shape(embedding_dim)
    y = torch.linspace(0, 1, steps=h)
    x = torch.linspace(0, 1, steps=w)
    YY, XX = torch.meshgrid(y, x, indexing="ij")
    pos = torch.stack([XX, YY], dim=-1).reshape(-1, 2)
    return torch.cdist(pos, pos, p=2)


# ───────────────── diff helpers ─────────────────


def grid_diffs(W: torch.Tensor) -> list[torch.Tensor]:
    out_feats, in_feats = W.shape
    h, w = get_grid_shape(out_feats)
    G = W.reshape(h, w, in_feats)
    diffs: list[torch.Tensor] = []
    if w > 1:
        diffs.append(G[:, :-1, :] - G[:, 1:, :])
    if h > 1:
        diffs.append(G[:-1, :, :] - G[1:, :, :])
    if h > 1 and w > 1:
        diffs.append(G[:-1, :-1, :] - G[1:, 1:, :])
        diffs.append(G[:-1, 1:, :] - G[1:, :-1, :])
    return diffs


def torus_diffs(W: torch.Tensor) -> list[torch.Tensor]:
    out_feats, in_feats = W.shape
    h, w = get_grid_shape(out_feats)
    G = W.reshape(h, w, in_feats)
    diffs: list[torch.Tensor] = []

    if w > 1:
        # Horizontal: interior + wrap (last col → first col)
        diffs.append(G[:, :-1, :] - G[:, 1:, :])  # (h, w-1, in_feats)
        diffs.append(G[:, -1, :] - G[:, 0, :])  # (h, in_feats)

    if h > 1:
        # Vertical: interior + wrap (last row → first row)
        diffs.append(G[:-1, :, :] - G[1:, :, :])  # (h-1, w, in_feats)
        diffs.append(G[-1, :, :] - G[0, :, :])  # (w, in_feats)

    if h > 1 and w > 1:
        # Diagonal bottom right: interior + right edge wrap + bottom edge wrap + corner wrap
        diffs.append(G[:-1, :-1, :] - G[1:, 1:, :])  # (h-1, w-1, in_feats)
        diffs.append(G[:-1, -1, :] - G[1:, 0, :])  # right edge → (h-1, in_feats)
        diffs.append(G[-1, :-1, :] - G[0, 1:, :])  # bottom edge → (w-1, in_feats)
        diffs.append(G[-1, -1, :] - G[0, 0, :])  # corner → (in_feats,)

        # Diagonal bottom left: interior + left edge wrap + bottom edge wrap + corner wrap
        diffs.append(G[:-1, 1:, :] - G[1:, :-1, :])  # (h-1, w-1, in_feats)
        diffs.append(G[:-1, 0, :] - G[1:, -1, :])  # left edge → (h-1, in_feats)
        diffs.append(G[-1, 1:, :] - G[0, :-1, :])  # bottom edge → (w-1, in_feats)
        diffs.append(G[-1, 0, :] - G[0, -1, :])  # corner → (in_feats,)

    return diffs


# ────────────── loss modules ──────────────


class Global_Topographic_Loss(nn.Module):
    """Global topographic regulariser on pre-activation features."""

    def __init__(self, weight: float = 1.0, emb_dim: int = 256) -> None:
        super().__init__()
        self.weight = weight
        self.D = pos_dist(emb_dim)

    def forward(self, pre_relu: torch.Tensor) -> torch.Tensor:
        if pre_relu.dim() != 2:
            raise ValueError(f"Expected 2-D [B,C], got {tuple(pre_relu.shape)}")
        self.D = self.D.to(pre_relu.device)
        _, n = pre_relu.shape
        Xn = F.normalize(pre_relu, p=2, dim=0, eps=1e-12)
        S = Xn.t() @ Xn
        i, j = torch.triu_indices(n, n, offset=1, device=pre_relu.device)
        d = self.D[i, j]
        s = S[i, j]
        loss = ((s - (1.0 / (d + 1.0))) ** 2).sum()
        return self.weight * (2.0 / (n * (n - 1))) * loss


class Local_WS_Loss(nn.Module):
    """Local weight-smoothing regulariser for a linear layer."""

    def __init__(self, weight: float = 1.0, topology: str = "grid") -> None:
        super().__init__()
        self.weight = weight
        if topology == "grid":
            self.diff_fn = grid_diffs
        elif topology == "torus":
            self.diff_fn = torus_diffs
        else:
            raise ValueError(f"Unknown topology: {topology}")

    def forward(self, linear_layer: nn.Linear | None = None) -> torch.Tensor:
        if linear_layer is None:
            raise ValueError("linear_layer must be provided.")
        W = linear_layer.weight
        diffs = self.diff_fn(W)
        if not diffs:
            return torch.zeros((), device=W.device, dtype=W.dtype)
        dists = [torch.linalg.norm(d, dim=-1) for d in diffs]
        return self.weight * torch.cat([x.reshape(-1) for x in dists]).mean()
