"""
Ensemble combination methods.
"""

from __future__ import annotations

from typing import List

import torch

METHODS = ["soft", "hard", "max_confidence", "conf_weighted"]


def combine_logits(
    logits_list: List[torch.Tensor], method: str = "soft"
) -> torch.Tensor:
    """
    Combine logits from multiple models.

    Args:
        logits_list: list of [N, C] tensors
        method: one of ``METHODS``

    Returns:
        [N, C] tensor of combined probabilities (or one-hot for hard).
    """
    logits_stack = torch.stack(logits_list)  # [M, N, C]
    N, C = logits_stack.shape[1], logits_stack.shape[2]
    probs = torch.softmax(logits_stack, dim=2)  # [M, N, C]

    if method == "soft":
        return probs.mean(dim=0)

    elif method == "hard":
        per_model_preds = logits_stack.argmax(dim=2)  # [M, N]
        hard_preds = torch.zeros(N, dtype=torch.long)
        # NOTE: This Python loop is intentionally kept for clarity.
        # Could be vectorized with torch.mode or scatter ops if N is very large,
        # but for typical eval sets (N ≤ 10k) the overhead is negligible.
        for i in range(N):
            votes = per_model_preds[:, i]
            counts = torch.bincount(votes, minlength=C)
            hard_preds[i] = counts.argmax()
        hard_onehot = torch.zeros(N, C)
        hard_onehot.scatter_(1, hard_preds.unsqueeze(1), 1.0)
        return hard_onehot

    elif method == "max_confidence":
        max_conf = probs.max(dim=2).values  # [M, N]
        best_idx = max_conf.argmax(dim=0)  # [N]
        idx_exp = best_idx.unsqueeze(0).unsqueeze(2).expand(1, N, C)
        return probs.gather(0, idx_exp).squeeze(0)

    elif method == "conf_weighted":
        confs = probs.max(dim=2).values  # [M, N]
        weights = confs / confs.sum(dim=0, keepdim=True)
        return torch.einsum("mn,mnc->nc", weights, probs)

    else:
        raise ValueError(f"Unknown method: {method}")


# NOTE: For ensemble hashing, use src.mlflow_utils.component_set_hash (canonical).
