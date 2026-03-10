"""
Profiling: Representational Dissimilarity Matrices.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pearson_corrcoef(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pearson correlation matrix for row-vectors in X [N, D] → [N, N]."""
    X = X.to(dtype=torch.float32, device="cpu")
    Xc = X - X.mean(dim=1, keepdim=True)
    norms = Xc.norm(dim=1, keepdim=True).clamp_min(eps)
    Y = Xc / norms
    return Y @ Y.t()


def pearson_rdm(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """1 − Pearson correlation → dissimilarity matrix."""
    rdm = 1.0 - pearson_corrcoef(X, eps)
    rdm.fill_diagonal_(0.0)
    return rdm


def upper_triangle_vector(M: torch.Tensor, include_diagonal: bool = True) -> torch.Tensor:
    """Upper-triangular values of a square matrix as a 1-D vector."""
    n = M.size(0)
    offset = 0 if include_diagonal else 1
    idx = torch.triu_indices(n, n, offset=offset)
    return M[idx[0], idx[1]].to(dtype=torch.float32, device="cpu")


def compute_embeddings(
    encoder: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """Run images through encoder in batches → [N, D] float32 CPU."""
    encoder.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, images.size(0), batch_size):
            batch = images[i : i + batch_size].to(device, non_blocking=True)
            out = encoder(batch)
            if out.ndim > 2:
                out = out.flatten(1)
            feats.append(out.detach().cpu().float())
    return torch.cat(feats)
