from __future__ import annotations

import numpy as np
import torch

# ─── HYBRID PROFILE MASKS ───

HYBRID_MASKS = {
    "hybrid_trueclass_argmaxlogits": ("true_class", "argmax_logits"),
    "hybrid_trueclass_argmaxsimilarity": ("true_class", "argmax_similarity"),
}


def apply_profile_mask(
    P: torch.Tensor,
    mask_type: str,
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

    if not torch.isfinite(P).all():
        raise ValueError("Profile tensor P contains NaN/Inf values")

    if mask_type == "true_class":
        mask = torch.ones(N, C, dtype=torch.bool, device=P.device)
        mask[torch.arange(N), labels] = False
        mask_3d = mask.unsqueeze(1).expand(N, M, C)
        return P[mask_3d].view(N, M, C - 1)

    elif mask_type == "argmax_similarity":
        mean_similarity = P.mean(dim=1)
        preds = mean_similarity.argmax(dim=1)
        mask = torch.ones(N, M, C, dtype=torch.bool, device=P.device)
        mask[
            torch.arange(N).unsqueeze(1),
            torch.arange(M).unsqueeze(0),
            preds.unsqueeze(1),
        ] = False
        return P[mask].view(N, M, C - 1)

    elif mask_type == "argmax_logits":
        if any(p is None for p in component_logit_preds):
            raise ValueError(
                "Cannot use argmax_logits profile mask because not all components have cached logits."
            )

        if any(
            not torch.isfinite(p).all() for p in component_logit_preds if p is not None
        ):
            raise ValueError(
                "Cannot use argmax_logits profile mask because logits contain NaN/Inf values."
            )

        if indices is not None:
            mean_logits = (
                torch.stack([p[indices] for p in component_logit_preds], dim=1)
                .to(P.device)
                .mean(dim=1)
            )
        else:
            mean_logits = (
                torch.stack(component_logit_preds, dim=1).to(P.device).mean(dim=1)
            )
        preds = mean_logits.argmax(dim=1)
        mask = torch.ones(N, M, C, dtype=torch.bool, device=P.device)
        mask[
            torch.arange(N).unsqueeze(1),
            torch.arange(M).unsqueeze(0),
            preds.unsqueeze(1),
        ] = False
        return P[mask].view(N, M, C - 1)

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

    corr = torch.bmm(P_n, P_n.transpose(1, 2))
    rdm = 1.0 - corr

    K = P_n.size(1)
    idx = torch.triu_indices(K, K, offset=1)
    return rdm[:, idx[0], idx[1]]


def assert_valid_feature_tensor(
    name: str, tensor: torch.Tensor, expected_rows: int
) -> None:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={tuple(tensor.shape)}")
    if tensor.shape[0] != expected_rows:
        raise ValueError(
            f"{name} row mismatch: expected {expected_rows}, got {tensor.shape[0]}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN/Inf values")
