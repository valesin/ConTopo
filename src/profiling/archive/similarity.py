"""
Profiling: similarity / cosine profiles between model representations.
"""

from __future__ import annotations

import torch
from src.profiling.rdm import pearson_corrcoef, upper_triangle_vector


def cosine_similarity_matrix(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Cosine similarity matrix for row-vectors in X [N, D] → [N, N]."""
    X = X.float()
    norms = X.norm(dim=1, keepdim=True).clamp_min(eps)
    Xn = X / norms
    return Xn @ Xn.t()


def rsa_correlation(rdm_a: torch.Tensor, rdm_b: torch.Tensor) -> float:
    """
    Representational Similarity Analysis: Pearson correlation between
    upper-triangular vectors of two RDMs.
    """
    va = upper_triangle_vector(rdm_a, include_diagonal=False)
    vb = upper_triangle_vector(rdm_b, include_diagonal=False)
    return float(pearson_corrcoef(torch.stack([va, vb]))[0, 1].item())
