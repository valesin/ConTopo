"""
Representational Dissimilarity Matrices and RSA.

Provides:
  - ``pearson_rdm``:       1 − Pearson correlation → [N, N] dissimilarity matrix
  - ``pearson_corrcoef``:  Pearson correlation matrix for row-vectors
  - ``upper_triangle_vector``: Extract upper-triangle as 1-D vector
  - ``rsa_correlation``:   Pearson correlation between upper triangles of two RDMs
"""

from __future__ import annotations

import torch


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


def rsa_correlation(rdm_a: torch.Tensor, rdm_b: torch.Tensor) -> float:
    """
    Representational Similarity Analysis: Pearson correlation between
    upper-triangular vectors of two RDMs.
    """
    va = upper_triangle_vector(rdm_a, include_diagonal=False)
    vb = upper_triangle_vector(rdm_b, include_diagonal=False)
    return float(pearson_corrcoef(torch.stack([va, vb]))[0, 1].item())
