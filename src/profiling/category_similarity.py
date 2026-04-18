"""
Category Similarity Profile computation.

Given model embeddings and anchor points, compute per-sample similarity
profiles using a specified metric (cosine or L2).

A **category similarity profile** for sample *i* is a vector of similarities
between sample *i*'s embedding and each anchor embedding:

    profile_i = [sim(e_i, a_0), sim(e_i, a_1), ..., sim(e_i, a_{K-1})]

where K = per_class × num_classes.

Supported metrics:
  - ``cosine``:  cosine similarity (higher = more similar)
  - ``l2``:      negative L2 distance  (higher = more similar, negated so
                 that the sign convention matches cosine)
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


def compute_similarity_profile(
    embeddings: torch.Tensor,
    anchor_embeddings: torch.Tensor,
    num_classes: int,
    metric: Literal["cosine", "l2"] = "cosine",
) -> torch.Tensor:
    """
    Compute similarity profile for every sample against all anchors.

    Args:
        embeddings:        [N, D] — sample embeddings.
        anchor_embeddings: [K, D] — anchor embeddings.
        num_classes:       Number of classes in the dataset.
        metric:            ``"cosine"`` or ``"l2"``.

    Returns:
        [N, num_classes] similarity profile tensor (float32).
    """
    embeddings = embeddings.float()
    anchor_embeddings = anchor_embeddings.float()

    if metric == "cosine":
        # Normalise then matmul → [N, K] cosine similarities
        e_norm = F.normalize(embeddings, p=2, dim=1)
        a_norm = F.normalize(anchor_embeddings, p=2, dim=1)
        sims = e_norm @ a_norm.t()

    elif metric == "l2":
        # Negative pairwise L2 distance → [N, K]
        # ||e - a||^2 = ||e||^2 + ||a||^2 - 2 e·a
        e_sq = (embeddings**2).sum(dim=1, keepdim=True)  # [N, 1]
        a_sq = (anchor_embeddings**2).sum(dim=1).unsqueeze(0)  # [1, K]
        dist_sq = e_sq + a_sq - 2.0 * embeddings @ anchor_embeddings.t()
        dist_sq = dist_sq.clamp_min(0.0)
        sims = -dist_sq.sqrt()  # negate so higher = more similar

    else:
        raise ValueError(f"Unknown similarity metric '{metric}'. Supported: cosine, l2")

    N = sims.shape[0]
    A = sims.shape[1] // num_classes
    return sims.view(N, num_classes, A).mean(dim=2)
