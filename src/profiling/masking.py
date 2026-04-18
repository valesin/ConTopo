from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np
import torch

# ─── HYBRID PROFILE MASKS ───

ProfileMaskType: TypeAlias = Literal[
    "true_class",
    "argmax_similarity",
    "argmax_logits",
    "none",
    "hybrid_trueclass_argmaxlogits",
    "hybrid_trueclass_argmaxsimilarity",
]

HYBRID_MASKS: dict[
    Literal[
        "hybrid_trueclass_argmaxlogits",
        "hybrid_trueclass_argmaxsimilarity",
    ],
    tuple[Literal["true_class"], Literal["argmax_logits", "argmax_similarity"]],
] = {
    "hybrid_trueclass_argmaxlogits": ("true_class", "argmax_logits"),
    "hybrid_trueclass_argmaxsimilarity": ("true_class", "argmax_similarity"),
}


def apply_profile_mask(
    P: torch.Tensor,
    mask_type: ProfileMaskType,
    labels: torch.Tensor,
    component_logit_preds: list[torch.Tensor | None],
    indices: np.ndarray | None = None,
) -> torch.Tensor:
    """Apply a single profile mask to P (N, M, C) → (N, M, C-1) or (N, M, C).

    Removes one class dimension per sample before RDM computation so that the
    resulting features reflect inter-model agreement structure rather than
    raw class affinity.

    Modes:
      true_class          - removes the ground-truth class (requires labels;
                            causes label leakage if applied to holdout).
      argmax_similarity   - removes the class with highest mean similarity across
                            models (label-free, safe at inference time).
      argmax_logits       - removes the class with highest mean ensemble logit
                            (label-free, safe at inference time).
      none                - no masking; returns P unchanged.

    Hybrid modes (defined in HYBRID_MASKS) apply true_class to train/val and
    an argmax variant to holdout, eliminating leakage at evaluation while
    preserving a stronger training signal.
    """
    N, M, C = P.shape

    if not P.isfinite().all():
        raise ValueError("Profile tensor P contains NaN/Inf values")

    if mask_type == "true_class":
        mask_3d = P.new_ones((N, M, C)).bool()
        for i in range(N):
            mask_3d[i, :, int(labels[i].item())] = False
        return P[mask_3d].view(N, M, C - 1)

    elif mask_type == "argmax_similarity":
        mean_similarity = P.mean(dim=1)
        preds = mean_similarity.argmax(dim=1)
        mask_3d = P.new_ones((N, M, C)).bool()
        for i in range(N):
            cls_idx = int(preds[i].item())
            mask_3d[i, :, cls_idx] = False
        return P[mask_3d].view(N, M, C - 1)

    elif mask_type == "argmax_logits":
        if any(p is None for p in component_logit_preds):
            raise ValueError(
                "Cannot use argmax_logits profile mask because not all components have cached logits."
            )

        logits_preds = [p for p in component_logit_preds if p is not None]

        if any(not p.isfinite().all() for p in logits_preds):
            raise ValueError(
                "Cannot use argmax_logits profile mask because logits contain NaN/Inf values."
            )

        if indices is not None:
            selected_logits = [p[indices].to(P.device) for p in logits_preds]
        else:
            selected_logits = [p.to(P.device) for p in logits_preds]
        mean_logits = selected_logits[0].new_zeros(selected_logits[0].shape)
        for logits in selected_logits:
            mean_logits += logits
        mean_logits /= float(len(selected_logits))
        preds = mean_logits.argmax(dim=1)
        mask_3d = P.new_ones((N, M, C)).bool()
        for i in range(N):
            cls_idx = int(preds[i].item())
            mask_3d[i, :, cls_idx] = False
        return P[mask_3d].view(N, M, C - 1)

    elif mask_type == "none":
        return P

    else:
        raise ValueError(f"Unknown profile_mask: {mask_type}")


def compute_rdm_features(P_masked: torch.Tensor) -> torch.Tensor:
    """Compute upper-triangular RDM features from masked profiles.

    Args:
        P_masked: (N, M, C') tensor of masked profiles
    Returns:
        S: (N, K*(K-1)/2) tensor of pairwise dissimilarities
    """
    Pc = P_masked - P_masked.mean(dim=2, keepdim=True)
    P_norm = Pc.norm(dim=2, keepdim=True).clamp_min(1e-8)
    P_n = Pc / P_norm

    corr = P_n.bmm(P_n.transpose(1, 2))
    rdm = 1.0 - corr

    K = P_n.size(1)
    pair_count = K * (K - 1) // 2
    out = rdm.new_empty((rdm.size(0), pair_count))
    col = 0
    for i in range(K):
        for j in range(i + 1, K):
            out[:, col] = rdm[:, i, j]
            col += 1
    return out


def assert_valid_feature_tensor(
    name: str, tensor: torch.Tensor, expected_rows: int
) -> None:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={tuple(tensor.shape)}")
    if tensor.shape[0] != expected_rows:
        raise ValueError(
            f"{name} row mismatch: expected {expected_rows}, got {tensor.shape[0]}"
        )
    if not tensor.isfinite().all():
        raise ValueError(f"{name} contains NaN/Inf values")
