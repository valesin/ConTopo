"""
Pairwise classifier agreement and diversity metrics.

Optimized Implementation:
- Uses matrix multiplication for confusion count calculation (Vectorized).
- Naming Convention:
    matrix_<metric>(...) -> (R, R) numpy array
    group_<metric>(...)  -> float scalar
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import torch
from numpy.linalg import norm
import scipy.stats


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class AgreementCounts:
    """
    Stores the four confusion matrix components for a pair of classifiers.
    """
    both_correct: int           # N11
    both_incorrect: int         # N00
    a_correct_b_incorrect: int  # N10
    a_incorrect_b_correct: int  # N01

    @property
    def total(self) -> int:
        return (
            self.both_correct
            + self.both_incorrect
            + self.a_correct_b_incorrect
            + self.a_incorrect_b_correct
        )


# ---------------------------------------------------------------------------
# Core Logic: Vectorized Confusion Counts
# ---------------------------------------------------------------------------


def matrix_confusion_counts(
    preds_list: List[torch.Tensor],
    labels: torch.Tensor,
) -> np.ndarray:
    """
    Compute the 4 confusion counts for every pair of classifiers.

    OPTIMIZATION NOTE:
    ------------------
    Replaces nested loops (O(R^2)) with Matrix Multiplication.
    
    1. Construct correctness matrix C of shape (N_samples, R_classifiers).
    2. Perform dot products (C.T @ C) to instantly compute intersection counts.
    
    Returns:
        np.ndarray of shape (4, R, R)
    """
    R = len(preds_list)
    N = labels.numel()
    
    # 1. Stack predictions and compare with labels to get Boolean Correctness Matrix
    try:
        preds_stack = torch.stack(preds_list).to(labels.device)
    except Exception as e:
        raise ValueError("preds_list elements must be tensors of the same shape.") from e
        
    if preds_stack.shape[1] != N:
         raise ValueError(f"Predictions length ({preds_stack.shape[1]}) does not match labels length ({N})")

    # C_bool: (R, N) where True = Correct
    C_bool = (preds_stack == labels.unsqueeze(0))
    
    # Convert to float for matrix multiplication: (N, R)
    C = C_bool.T.float()
    
    # 2. Compute Intersection Counts via Matrix Multiplication
    # N11: Both correct (C_i * C_j) -> C.T @ C
    n11_mat = C.T @ C  # (R, R)
    
    # N00: Both incorrect ((1-C)_i * (1-C)_j) -> (1-C).T @ (1-C)
    C_inv = 1.0 - C
    n00_mat = C_inv.T @ C_inv
    
    # N10: A correct, B incorrect (C_i * (1-C)_j) -> C.T @ (1-C)
    n10_mat = C.T @ C_inv
    
    # N01: A incorrect, B correct ((1-C)_i * C_j) -> (1-C).T @ C
    n01_mat = C_inv.T @ C
    
    # 3. Pack into (4, R, R) numpy array
    counts_stack = np.zeros((4, R, R), dtype=int)
    counts_stack[0] = n11_mat.cpu().numpy().astype(int)
    counts_stack[1] = n00_mat.cpu().numpy().astype(int)
    counts_stack[2] = n10_mat.cpu().numpy().astype(int)
    counts_stack[3] = n01_mat.cpu().numpy().astype(int)
    
    return counts_stack


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _counts_from_stack(counts_stack: np.ndarray, i: int, j: int) -> AgreementCounts:
    """Extract an AgreementCounts object for pair (i, j) from the stack."""
    return AgreementCounts(
        both_correct=int(counts_stack[0, i, j]),
        both_incorrect=int(counts_stack[1, i, j]),
        a_correct_b_incorrect=int(counts_stack[2, i, j]),
        a_incorrect_b_correct=int(counts_stack[3, i, j]),
    )


def _apply_over_pairs(
    counts_stack: np.ndarray,
    fn: Callable[[AgreementCounts], float],
) -> np.ndarray:
    """Apply a metric function over all pairs in the counts stack."""
    R = counts_stack.shape[1]
    result = np.full((R, R), np.nan, dtype=float)
    for i in range(R):
        for j in range(R):
            result[i, j] = fn(_counts_from_stack(counts_stack, i, j))
    return result


def _compute_matrix_generic(
    items: List[Any],
    metric_fn: Callable[[Any, Any], float],
    **kwargs
) -> np.ndarray:
    """Generic helper for metrics that don't use the confusion counts stack (e.g. IoU)."""
    R = len(items)
    result = np.full((R, R), np.nan, dtype=float)
    for i in range(R):
        for j in range(R):
            val = metric_fn(items[i], items[j], **kwargs)
            if val is not None:
                result[i, j] = val
    return result


def _average_off_diagonal(mat: np.ndarray) -> float:
    """Compute mean of off-diagonal elements."""
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return float("nan")
    R = mat.shape[0]
    if R < 2:
        return float("nan")
    # Extract off-diagonal elements
    off_diag = mat[~np.eye(R, dtype=bool)]
    return float(np.nanmean(off_diag))


# ---------------------------------------------------------------------------
# Scalar Metric Logic (Math)
# ---------------------------------------------------------------------------


def _calc_asym_ratio(counts: AgreementCounts) -> float:
    if counts.total == 0: return float("nan")
    return counts.a_correct_b_incorrect / counts.total


def _calc_asym_ratio_reverse(counts: AgreementCounts) -> float:
    if counts.total == 0: return float("nan")
    return counts.a_incorrect_b_correct / counts.total


def _calc_correctness_disagreement(counts: AgreementCounts) -> float:
    if counts.total == 0: return float("nan")
    return (counts.a_correct_b_incorrect + counts.a_incorrect_b_correct) / counts.total


def _calc_error_conditional_disagreement(counts: AgreementCounts) -> float:
    denom = counts.a_correct_b_incorrect + counts.a_incorrect_b_correct + counts.both_incorrect
    if denom == 0: return float("nan")
    return (counts.a_correct_b_incorrect + counts.a_incorrect_b_correct) / denom


def _calc_overall_agreement(counts: AgreementCounts) -> float:
    if counts.total == 0: return float("nan")
    return (counts.both_correct + counts.both_incorrect) / counts.total


def _calc_jaccard(counts: AgreementCounts) -> float:
    union = counts.both_correct + counts.a_correct_b_incorrect + counts.a_incorrect_b_correct
    if union == 0: return float("nan")
    return counts.both_correct / union


def _calc_cohens_kappa(counts: AgreementCounts) -> float:
    total = counts.total
    if total == 0: return float("nan")
    pa = (counts.both_correct + counts.a_correct_b_incorrect) / total
    pb = (counts.both_correct + counts.a_incorrect_b_correct) / total
    pe = pa * pb + (1 - pa) * (1 - pb)
    po = (counts.both_correct + counts.both_incorrect) / total
    return (po - pe) / (1 - pe) if (1 - pe) != 0 else float("nan")


def _calc_double_fault(counts: AgreementCounts) -> float:
    if counts.total == 0: return float("nan")
    return counts.both_incorrect / counts.total


def _calc_q_statistic(counts: AgreementCounts) -> float:
    ad = counts.both_correct * counts.both_incorrect
    bc = counts.a_correct_b_incorrect * counts.a_incorrect_b_correct
    denom = ad + bc
    return float((ad - bc) / denom) if denom != 0 else float("nan")


def _calc_pred_disagreement(pred_a: torch.Tensor, pred_b: torch.Tensor) -> float:
    N = pred_a.numel()
    if N == 0: return float("nan")
    return (pred_a != pred_b).float().sum().item() / N


def _calc_output_correlation(probs_a: np.ndarray, probs_b: np.ndarray) -> float:
    flat_a = probs_a.ravel().astype(np.float64)
    flat_b = probs_b.ravel().astype(np.float64)
    if len(flat_a) < 2: return float("nan")
    return float(np.corrcoef(flat_a, flat_b)[0, 1])


def _calc_param_cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    if vec_a is None or vec_b is None: return float("nan")
    va = vec_a.detach().cpu().numpy() if hasattr(vec_a, 'detach') else vec_a
    vb = vec_b.detach().cpu().numpy() if hasattr(vec_b, 'detach') else vec_b
    
    denom = norm(va) * norm(vb)
    if denom == 0: return float("nan")
    return float(np.dot(va, vb) / denom)


def _calc_iou_top_n(logits_a: torch.Tensor, logits_b: torch.Tensor, n: int = 5) -> float:
    if logits_a is logits_b: return 1.0
    inds_a = logits_a.topk(n, dim=1).indices
    inds_b = logits_b.topk(n, dim=1).indices
    matches = (inds_a.unsqueeze(2) == inds_b.unsqueeze(1))
    inter = matches.sum(dim=(1, 2)).float()
    union = 2 * n - inter
    return (inter / union).mean().item()


# ---------------------------------------------------------------------------
# Public API: Matrix Metrics (RxR Matrix)
# ---------------------------------------------------------------------------


def matrix_asym_ratio(counts_stack: np.ndarray) -> np.ndarray:
    return _apply_over_pairs(counts_stack, _calc_asym_ratio)


def matrix_asym_ratio_reverse(counts_stack: np.ndarray) -> np.ndarray:
    return _apply_over_pairs(counts_stack, _calc_asym_ratio_reverse)


def matrix_correctness_disagreement(counts_stack: np.ndarray) -> np.ndarray:
    return _apply_over_pairs(counts_stack, _calc_correctness_disagreement)


def matrix_error_conditional_disagreement(counts_stack: np.ndarray) -> np.ndarray:
    return _apply_over_pairs(counts_stack, _calc_error_conditional_disagreement)


def matrix_overall_agreement(counts_stack: np.ndarray) -> np.ndarray:
    return _apply_over_pairs(counts_stack, _calc_overall_agreement)


def matrix_cohens_kappa(counts_stack: np.ndarray) -> np.ndarray:
    return _apply_over_pairs(counts_stack, _calc_cohens_kappa)


def matrix_jaccard(counts_stack: np.ndarray) -> np.ndarray:
    return _apply_over_pairs(counts_stack, _calc_jaccard)


def matrix_double_fault(counts_stack: np.ndarray) -> np.ndarray:
    return _apply_over_pairs(counts_stack, _calc_double_fault)


def matrix_q_statistic(counts_stack: np.ndarray) -> np.ndarray:
    return _apply_over_pairs(counts_stack, _calc_q_statistic)


def matrix_pred_disagreement(preds_list: List[torch.Tensor]) -> np.ndarray:
    return _compute_matrix_generic(preds_list, _calc_pred_disagreement)


def matrix_output_correlation(probs_list: List[np.ndarray]) -> np.ndarray:
    return _compute_matrix_generic(probs_list, _calc_output_correlation)


def matrix_param_cosine(param_vecs: List[np.ndarray]) -> np.ndarray:
    return _compute_matrix_generic(param_vecs, _calc_param_cosine)


def matrix_iou_top_n(logits_list: List[torch.Tensor], n: int = 5) -> np.ndarray:
    return _compute_matrix_generic(logits_list, _calc_iou_top_n, n=n)


# ---------------------------------------------------------------------------
# Public API: Group Metrics (Scalar Average)
# ---------------------------------------------------------------------------


def group_asym_ratio(counts_stack: np.ndarray) -> float:
    return _average_off_diagonal(matrix_asym_ratio(counts_stack))


def group_asym_ratio_reverse(counts_stack: np.ndarray) -> float:
    return _average_off_diagonal(matrix_asym_ratio_reverse(counts_stack))


def group_correctness_disagreement(counts_stack: np.ndarray) -> float:
    return _average_off_diagonal(matrix_correctness_disagreement(counts_stack))


def group_error_conditional_disagreement(counts_stack: np.ndarray) -> float:
    return _average_off_diagonal(matrix_error_conditional_disagreement(counts_stack))


def group_overall_agreement(counts_stack: np.ndarray) -> float:
    return _average_off_diagonal(matrix_overall_agreement(counts_stack))


def group_cohens_kappa(counts_stack: np.ndarray) -> float:
    return _average_off_diagonal(matrix_cohens_kappa(counts_stack))


def group_jaccard(counts_stack: np.ndarray) -> float:
    return _average_off_diagonal(matrix_jaccard(counts_stack))


def group_double_fault(counts_stack: np.ndarray) -> float:
    return _average_off_diagonal(matrix_double_fault(counts_stack))


def group_q_statistic(counts_stack: np.ndarray) -> float:
    return _average_off_diagonal(matrix_q_statistic(counts_stack))


def group_pred_disagreement(preds_list: List[torch.Tensor]) -> float:
    return _average_off_diagonal(matrix_pred_disagreement(preds_list))


def group_output_correlation(probs_list: List[np.ndarray]) -> float:
    return _average_off_diagonal(matrix_output_correlation(probs_list))


def group_param_cosine(param_vecs: List[np.ndarray]) -> float:
    return _average_off_diagonal(matrix_param_cosine(param_vecs))


def group_iou_top_n(logits_list: List[torch.Tensor], n: int = 5) -> float:
    return _average_off_diagonal(matrix_iou_top_n(logits_list, n=n))