"""
Pairwise classifier agreement and diversity metrics.

Provides individual functions, each computing one metric over all pairs of
classifiers.  Most metrics operate on a shared *counts_stack* (4, R, R)
produced once by ``pairwise_confusion_counts``, where the four slices are:

    0 — both correct   (N11)
    1 — both incorrect  (N00)
    2 — A correct, B incorrect (N01)
    3 — A incorrect, B correct (N10)
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import torch
from numpy.linalg import norm


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AgreementCounts:
    both_correct: int
    both_incorrect: int
    a_correct_b_incorrect: int
    a_incorrect_b_correct: int

    @property
    def total(self) -> int:
        return (
            self.both_correct
            + self.both_incorrect
            + self.a_correct_b_incorrect
            + self.a_incorrect_b_correct
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_shapes(*arrays: torch.Tensor) -> None:
    if not arrays:
        raise ValueError("_validate_shapes requires at least one tensor")
    shapes = {tuple(a.shape) for a in arrays}
    if len(shapes) != 1:
        raise ValueError(
            f"Input tensors must share the same shape; got shapes: {shapes}"
        )


def _counts_from_stack(counts_stack: np.ndarray, i: int, j: int) -> AgreementCounts:
    """Extract an AgreementCounts for pair (i, j) from a (4, R, R) stack."""
    return AgreementCounts(
        both_correct=int(counts_stack[0, i, j]),
        both_incorrect=int(counts_stack[1, i, j]),
        a_correct_b_incorrect=int(counts_stack[2, i, j]),
        a_incorrect_b_correct=int(counts_stack[3, i, j]),
    )



def _apply_over_pairs(
    counts_stack: np.ndarray,
    fn,
    **kwargs
) -> np.ndarray:
    """Apply a single-pair metric function over all pairs in a counts stack."""
    R = counts_stack.shape[1]
    result = np.full((R, R), np.nan, dtype=float)
    for i in range(R):
        for j in range(R):
            result[i, j] = fn(_counts_from_stack(counts_stack, i, j), **kwargs)
    return result


def compute_pairwise_metric(
    items: List[Any],
    metric_fn: Callable[[Any, Any], float],
    **kwargs
) -> np.ndarray:
    """
    Generic helper to compute a metric over all pairs of items.
    
    Args:
        items: List of R items (tensors, arrays, etc.)
        metric_fn: Function taking (item_a, item_b, **kwargs) -> float
    
    Returns:
        (R, R) numpy array.
    """
    R = len(items)
    result = np.full((R, R), np.nan, dtype=float)
    for i in range(R):
        for j in range(R):
            val = metric_fn(items[i], items[j], **kwargs)
            if val is not None:
                result[i, j] = val
    return result


def pairwise(
    data: Union[List[Any], np.ndarray],
    metric_fn: Callable,
    **kwargs
) -> np.ndarray:
    """
    Compute a pairwise metric over the given data.

    Args:
        data: Either a list of items (for metrics like pred_disagreement)
              or a (4, R, R) numpy array of confusion counts (for metrics like kohen's kappa).
        metric_fn: The single-pair metric function to apply.
        **kwargs: Additional arguments passed to metric_fn.

    Returns:
        (R, R) numpy array.
    """
    if isinstance(data, np.ndarray) and data.ndim == 3 and data.shape[0] == 4:
        # Assume it's a counts stack
        return _apply_over_pairs(data, metric_fn, **kwargs)
    elif isinstance(data, (list, tuple)):
        return compute_pairwise_metric(data, metric_fn, **kwargs)
    else:
        raise ValueError(
            "Data must be either a list of items or a (4, R, R) counts stack."
        )


def _pearsonr_from_flat(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r for 1-D numpy arrays; returns NaN on zero std."""
    if x.size == 0 or y.size == 0 or x.shape != y.shape:
        return float("nan")
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    xm = x.mean()
    ym = y.mean()
    xm_d = x - xm
    ym_d = y - ym
    denom = np.sqrt((xm_d**2).sum() * (ym_d**2).sum())
    if denom == 0:
        return float("nan")
    return float((xm_d * ym_d).sum() / denom)


def _average_off_diagonal(mat: np.ndarray) -> float:
    """Compute mean of off-diagonal elements in a square matrix."""
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return float("nan")
    R = mat.shape[0]
    if R < 2:
        return float("nan")
    # Mask diagonal
    off_diag = mat[~np.eye(R, dtype=bool)]
    return float(np.nanmean(off_diag))


# ---------------------------------------------------------------------------
# Single-pair metric functions
# ---------------------------------------------------------------------------


def compute_confusion_counts(
    correct_a: torch.Tensor, correct_b: torch.Tensor
) -> AgreementCounts:
    """
    Compute the four canonical counts between two binary correctness vectors.

    Args:
        correct_a: boolean Tensor (N,) where True indicates classifier A is correct.
        correct_b: boolean Tensor (N,) where True indicates classifier B is correct.

    Returns:
        AgreementCounts with integer fields.
    """
    if not isinstance(correct_a, torch.Tensor) or not isinstance(
        correct_b, torch.Tensor
    ):
        raise TypeError("correct_a and correct_b must be torch.Tensor")
    if correct_a.ndim != 1 or correct_b.ndim != 1:
        raise ValueError("correct_a and correct_b must be 1-D tensors")
    if correct_a.numel() != correct_b.numel():
        raise ValueError("correct_a and correct_b must have equal length")

    a_mask = torch.isfinite(correct_a) & (correct_a != 0)
    b_mask = torch.isfinite(correct_b) & (correct_b != 0)

    return AgreementCounts(
        both_correct=int(torch.logical_and(a_mask, b_mask).sum().item()),
        both_incorrect=int(torch.logical_and(~a_mask, ~b_mask).sum().item()),
        a_correct_b_incorrect=int(torch.logical_and(a_mask, ~b_mask).sum().item()),
        a_incorrect_b_correct=int(torch.logical_and(~a_mask, b_mask).sum().item()),
    )


def asym_ratio(counts: AgreementCounts) -> float:
    """A_correct_B_incorrect / N.  Returns NaN when total == 0."""
    total = counts.total
    if total == 0:
        return float("nan")
    return counts.a_correct_b_incorrect / total


def asym_ratio_reverse(counts: AgreementCounts) -> float:
    """A_incorrect_B_correct / N."""
    total = counts.total
    if total == 0:
        return float("nan")
    return counts.a_incorrect_b_correct / total


def correctness_disagreement_ratio(counts: AgreementCounts) -> float:
    """
    (A_correct_B_incorrect + A_incorrect_B_correct) / N.

    Fraction of samples where exactly one classifier is correct.
    Does NOT count samples where both are wrong but predict different classes.
    """
    total = counts.total
    if total == 0:
        return float("nan")
    return (counts.a_correct_b_incorrect + counts.a_incorrect_b_correct) / total


def error_conditional_disagreement(counts: AgreementCounts) -> float:
    """
    (N01 + N10) / (N01 + N10 + N00).

    Among samples where at least one classifier is wrong, fraction where
    exactly one is wrong.  Returns NaN when both classifiers are perfect.
    """
    numerator = counts.a_correct_b_incorrect + counts.a_incorrect_b_correct
    denominator = numerator + counts.both_incorrect
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def overall_agreement_rate(counts: AgreementCounts) -> float:
    """Fraction where both are correct or both are incorrect."""
    total = counts.total
    if total == 0:
        return float("nan")
    return (counts.both_correct + counts.both_incorrect) / total


def jaccard_correct_sets(counts: AgreementCounts) -> float:
    """Jaccard index over the sets of samples classified correctly."""
    inter = counts.both_correct
    union = (
        counts.both_correct
        + counts.a_correct_b_incorrect
        + counts.a_incorrect_b_correct
    )
    if union == 0:
        return float("nan")
    return inter / union


def cohens_kappa(counts: AgreementCounts) -> float:
    """Cohen's kappa for binary correctness agreement."""
    total = counts.total
    if total == 0:
        return float("nan")
    pa_correct = (counts.both_correct + counts.a_correct_b_incorrect) / total
    pb_correct = (counts.both_correct + counts.a_incorrect_b_correct) / total
    po = (counts.both_correct + counts.both_incorrect) / total
    pe = pa_correct * pb_correct + (1.0 - pa_correct) * (1.0 - pb_correct)
    denom = 1.0 - pe
    if denom == 0:
        return float("nan")
    return (po - pe) / denom


def _binomial_two_sided_pvalue(b: int, c: int) -> float:
    """Exact two-sided p-value for McNemar's test (binomial)."""
    n = b + c
    if n == 0:
        return float("nan")
    try:
        import scipy.stats as _ss

        if hasattr(_ss, "binomtest"):
            return float(
                _ss.binomtest(k=min(b, c), n=n, p=0.5, alternative="two-sided").pvalue
            )
        else:
            return float(_ss.binom_test(min(b, c), n=n, p=0.5, alternative="two-sided"))
    except Exception:
        tail = 0.0
        k_max = min(b, c)
        p_single = 0.5**n
        for k in range(0, k_max + 1):
            tail += comb(n, k) * p_single
        return float(min(1.0, 2.0 * tail))


def mcnemar_pvalue(counts: AgreementCounts) -> float:
    """McNemar's exact two-sided p-value."""
    return _binomial_two_sided_pvalue(
        counts.a_correct_b_incorrect, counts.a_incorrect_b_correct
    )


def pred_disagreement(pred_a: torch.Tensor, pred_b: torch.Tensor) -> float:
    """
    Fraction of samples where predicted labels differ.

    Captures both "one correct / one wrong" AND "both wrong, different class".
    Does not require ground-truth labels.
    """
    if pred_a.shape != pred_b.shape:
        raise ValueError("Predictions must have the same shape.")
    
    N = pred_a.numel()
    if N == 0:
        return float("nan")
        
    diff = (pred_a != pred_b).to(torch.float32).sum().item()
    return diff / float(N)


def norm_pred_disagreement(
    disagreement: float, ensemble_acc: float
) -> float:
    """
    Prediction disagreement normalised by ensemble error rate.
    
    val = disagreement / (1 - ensemble_accuracy)
    """
    denom = 1.0 - ensemble_acc
    if denom == 0.0:
        return float("nan")
    return disagreement / denom


def double_fault(counts: AgreementCounts, N: int) -> float:
    """Fraction of samples where both classifiers are wrong."""
    if N == 0:
        return float("nan")
    return counts.both_incorrect / float(N)


def output_correlation(probs_a: np.ndarray, probs_b: np.ndarray) -> float:
    """Pearson correlation over flattened probability vectors."""
    return _pearsonr_from_flat(probs_a.ravel(), probs_b.ravel())


def q_statistic(counts: AgreementCounts) -> float:
    """Yule's Q statistic: (N11*N00 - N01*N10) / (N11*N00 + N01*N10)."""
    N11 = counts.both_correct
    N00 = counts.both_incorrect
    N01 = counts.a_correct_b_incorrect
    N10 = counts.a_incorrect_b_correct
    denom = N11 * N00 + N01 * N10
    if denom == 0:
        return float("nan")
    return float((N11 * N00 - N01 * N10) / denom)


def param_cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between flattened parameter vectors."""
    if vec_a is None or vec_b is None:
        return float("nan")
        
    # Standardize input to numpy
    if hasattr(vec_a, "detach"):
        vec_a = vec_a.detach().cpu().numpy()
    if hasattr(vec_b, "detach"):
        vec_b = vec_b.detach().cpu().numpy()
        
    denom = float(norm(vec_a) * norm(vec_b))
    if denom == 0.0:
        return float("nan")
        
    return float(np.dot(vec_a, vec_b) / denom)


def iou_top_n_single(
    logits_a: torch.Tensor, logits_b: torch.Tensor, n: int = 5
) -> float:
    """
    Compute average Top-N Intersection over Union (IoU) for a pair of models.
    """
    if logits_a is logits_b:
        return 1.0
        
    # top_indices: [N, n]
    inds_a = logits_a.topk(n, dim=1).indices
    inds_b = logits_b.topk(n, dim=1).indices
    
    # Expand for broadcasting: [N, n, 1] vs [N, 1, n]
    matches = (inds_a.unsqueeze(2) == inds_b.unsqueeze(1))
    
    # intersection size per sample: sum over n,n dims -> [N]
    inter = matches.sum(dim=(1, 2)).float()
    
    # Union size = |A| + |B| - |A n B| = n + n - inter
    union = 2 * n - inter
    
    # Avoid division by zero (though union should be >= n >= 1 usually)
    iou = inter / union
    return iou.mean().item()


# ---------------------------------------------------------------------------
# Pairwise metric functions  (each returns an R×R matrix)
# ---------------------------------------------------------------------------


def pairwise_confusion_counts(
    preds_list: List[torch.Tensor],
    labels: torch.Tensor,
) -> np.ndarray:
    """
    Compute the 4 confusion counts for every pair of classifiers.

    Args:
        preds_list: list of R 1-D prediction tensors (each shape [N]).
        labels: 1-D ground-truth tensor (shape [N]).

    Returns:
        np.ndarray of shape (4, R, R) with slices
        [N11, N00, N01, N10] (see module docstring).
    """
    R = len(preds_list)
    N = labels.numel()
    for p in preds_list:
        if p.ndim != 1 or p.numel() != N:
            raise ValueError("All preds must be 1-D tensors of same length as labels")

    counts_stack = np.zeros((4, R, R), dtype=int)
    for i in range(R):
        for j in range(R):
            a_correct = preds_list[i] == labels
            b_correct = preds_list[j] == labels
            c = compute_confusion_counts(
                a_correct.to(torch.float32), b_correct.to(torch.float32)
            )
            counts_stack[0, i, j] = c.both_correct
            counts_stack[1, i, j] = c.both_incorrect
            counts_stack[2, i, j] = c.a_correct_b_incorrect
            counts_stack[3, i, j] = c.a_incorrect_b_correct
    return counts_stack


def pairwise_asym_ratio(counts_stack: np.ndarray) -> np.ndarray:
    """A_correct_B_incorrect / N for every pair."""
    return _apply_over_pairs(counts_stack, asym_ratio)


def pairwise_asym_ratio_reverse(counts_stack: np.ndarray) -> np.ndarray:
    """A_incorrect_B_correct / N for every pair."""
    return _apply_over_pairs(counts_stack, asym_ratio_reverse)


def pairwise_correctness_disagreement(counts_stack: np.ndarray) -> np.ndarray:
    """Fraction where exactly one classifier is correct."""
    return _apply_over_pairs(counts_stack, correctness_disagreement_ratio)


def pairwise_error_conditional_disagreement(counts_stack: np.ndarray) -> np.ndarray:
    """(N01+N10)/(N01+N10+N00) for every pair."""
    return _apply_over_pairs(counts_stack, error_conditional_disagreement)


def pairwise_overall_agreement(counts_stack: np.ndarray) -> np.ndarray:
    """Fraction where both agree (both correct or both wrong)."""
    return _apply_over_pairs(counts_stack, overall_agreement_rate)


def pairwise_cohens_kappa(counts_stack: np.ndarray) -> np.ndarray:
    """Cohen's kappa for every pair."""
    return _apply_over_pairs(counts_stack, cohens_kappa)


def pairwise_jaccard(counts_stack: np.ndarray) -> np.ndarray:
    """Jaccard index over correct-sets for every pair."""
    return _apply_over_pairs(counts_stack, jaccard_correct_sets)


def pairwise_mcnemar_p(counts_stack: np.ndarray) -> np.ndarray:
    """McNemar two-sided p-value for every pair."""
    return _apply_over_pairs(counts_stack, mcnemar_pvalue)


def pairwise_pred_disagreement(preds_list: List[torch.Tensor]) -> np.ndarray:
    """
    Fraction of samples where predicted labels differ.

    Captures both "one correct / one wrong" AND "both wrong, different class".
    Does not require ground-truth labels.
    """
    return compute_pairwise_metric(preds_list, pred_disagreement)


def pairwise_norm_pred_disagreement(
    pred_disagreement_mat: np.ndarray,
    ensemble_probs: Dict[Union[str, Tuple[int, int]], Dict[str, torch.Tensor]],
    labels: torch.Tensor,
) -> Dict[str, np.ndarray]:
    """
    Prediction disagreement normalised by ensemble error rate.

    For each ensemble method (soft, hard, max_confidence, conf_weighted),
    computes::

        norm_dis[i, j] = pred_disagreement[i, j] / (1 - ensemble_accuracy)

    Args:
        pred_disagreement_mat: (R, R) matrix from ``pairwise_pred_disagreement``.
        ensemble_probs: dict from ``compute_all_ensemble_probs``, keyed by
            ``(i, j)`` tuples (i < j).  Each value maps method name to a
            probability tensor of shape [N, C].
        labels: 1-D ground-truth tensor (shape [N]).

    Returns:
        Dict mapping method name → (R, R) matrix.
    """
    R = pred_disagreement_mat.shape[0]
    # Discover ensemble methods from the first pairwise entry
    pair_keys = [k for k in ensemble_probs if k != "all"]
    if not pair_keys:
        return {}
    methods = list(ensemble_probs[pair_keys[0]].keys())

    result: Dict[str, np.ndarray] = {}
    for method in methods:
        mat = np.full((R, R), np.nan, dtype=float)
        for i in range(R):
            for j in range(R):
                if i == j:
                    mat[i, j] = 0.0
                    continue
                # ensemble_probs only stores (min, max) pairs
                key = (min(i, j), max(i, j))
                if key not in ensemble_probs:
                    continue
                
                probs_t = ensemble_probs[key][method]
                ens_pred = probs_t.argmax(dim=1)
                ens_acc = float((ens_pred == labels).to(torch.float32).mean().item())
                
                # Use the single-pair function
                mat[i, j] = norm_pred_disagreement(pred_disagreement_mat[i, j], ens_acc)
        result[method] = mat
    return result


def pairwise_double_fault(counts_stack: np.ndarray, N: int) -> np.ndarray:
    """Fraction of samples where both classifiers are wrong."""
    return _apply_over_pairs(counts_stack, lambda c: double_fault(c, N))


def pairwise_output_correlation(probs_list: List[np.ndarray]) -> np.ndarray:
    """Pearson correlation over flattened probability vectors."""
    return compute_pairwise_metric(probs_list, output_correlation)


def pairwise_q_statistic(counts_stack: np.ndarray) -> np.ndarray:
    """Yule's Q statistic: (N11*N00 - N01*N10) / (N11*N00 + N01*N10)."""
    return _apply_over_pairs(counts_stack, q_statistic)


def pairwise_param_cosine(param_vecs: List[np.ndarray]) -> np.ndarray:
    """Cosine similarity between flattened parameter vectors."""
    return compute_pairwise_metric(param_vecs, param_cosine)


def pairwise_iou_top_n(
    logits_list: List[torch.Tensor], n: int = 5
) -> np.ndarray:
    """
    Compute pairwise average Top-N Intersection over Union (IoU).

    For each sample, get the set of top-N predicted class indices for both models.
    Compute sizes of intersection and union, then IoU. Average over samples.

    Args:
        logits_list: list of R logits tensors (shape [N, C]).
        n: number of top predictions to consider.

    Returns:
        np.ndarray of shape (R, R) with average IoU values.
    """
    return compute_pairwise_metric(logits_list, iou_top_n_single, n=n)


def group_confusion_counts(preds_list: List[torch.Tensor], labels: torch.Tensor) -> float:
    """
    Placeholder for group confusion counts.
    Returns average 'both_correct' count over all pairs (not really a standard metric).
    """
    counts_stack = pairwise_confusion_counts(preds_list, labels)
    # Return average N11 (both correct) as a placeholder scalar?
    # Or just NaN as it's ambiguous.
    return float("nan")


def group_asym_ratio(counts_stack: np.ndarray) -> float:
    """Average pairwise asymmetric ratio."""
    return _average_off_diagonal(pairwise_asym_ratio(counts_stack))


def group_correctness_disagreement(counts_stack: np.ndarray) -> float:
    """Average pairwise correctness disagreement."""
    return _average_off_diagonal(pairwise_correctness_disagreement(counts_stack))


def group_error_conditional_disagreement(counts_stack: np.ndarray) -> float:
    """Average pairwise error-conditional disagreement."""
    return _average_off_diagonal(pairwise_error_conditional_disagreement(counts_stack))


def group_overall_agreement(counts_stack: np.ndarray) -> float:
    """Average pairwise overall agreement."""
    return _average_off_diagonal(pairwise_overall_agreement(counts_stack))


def group_cohens_kappa(counts_stack: np.ndarray) -> float:
    """Average pairwise Cohen's Kappa."""
    return _average_off_diagonal(pairwise_cohens_kappa(counts_stack))


def group_jaccard(counts_stack: np.ndarray) -> float:
    """Average pairwise Jaccard index."""
    return _average_off_diagonal(pairwise_jaccard(counts_stack))


def group_mcnemar_p(counts_stack: np.ndarray) -> float:
    """Average pairwise McNemar's test p-value."""
    return _average_off_diagonal(pairwise_mcnemar_p(counts_stack))


def group_pred_disagreement(preds_list: List[torch.Tensor]) -> float:
    """Average pairwise prediction disagreement."""
    return _average_off_diagonal(pairwise_pred_disagreement(preds_list))


def group_norm_pred_disagreement(
    pred_disagreement: np.ndarray,
    ensemble_probs: Dict[Union[str, Tuple[int, int]], Dict[str, torch.Tensor]],
    labels: torch.Tensor,
) -> Dict[str, float]:
    """
    Average pairwise normalized prediction disagreement.
    Returns a dict mapping ensemble method -> average metric value.
    """
    pairwise_dict = pairwise_norm_pred_disagreement(
        pred_disagreement, ensemble_probs, labels
    )
    result = {}
    for method, mat in pairwise_dict.items():
        result[method] = _average_off_diagonal(mat)
    return result


def group_double_fault(counts_stack: np.ndarray, N: int) -> float:
    """Average pairwise double fault."""
    return _average_off_diagonal(pairwise_double_fault(counts_stack, N))


def group_output_correlation(probs_list: List[np.ndarray]) -> float:
    """Average pairwise output correlation."""
    return _average_off_diagonal(pairwise_output_correlation(probs_list))


def group_q_statistic(counts_stack: np.ndarray) -> float:
    """
    Average pairwise Yule's Q statistic.
    
    Formula: 2/(L(L-1)) * sum_{i<j} Q_{i,j}
    """
    return _average_off_diagonal(pairwise_q_statistic(counts_stack))


def group_param_cosine(param_vecs: List[np.ndarray]) -> float:
    """Average pairwise parameter cosine similarity."""
    return _average_off_diagonal(pairwise_param_cosine(param_vecs))


def group_iou_top_n(logits_list: List[torch.Tensor], n: int = 5) -> float:
    """Average pairwise Top-N IoU."""
    return _average_off_diagonal(pairwise_iou_top_n(logits_list, n=n))


def group_generalized_diversity(
    preds_list: List[torch.Tensor], labels: torch.Tensor
) -> float:
    """
    Generalized Diversity (GD) for an ensemble.
    GD = 1 - (p(2 failures) / p(1 failure))
    
    This implementation follows the Kuncheva definition where:
    p(1) = probability that at least one classifier fails.
    p(2) = probability that two randomly chosen classifiers fail.
    
    However, strictly speaking, GD is often defined via failure probabilities.
    Here we implement a placeholder based on the pairwise double fault average?
    

    If we interpret p(2 failures) as the average pairwise double fault (N00/N),
    and p(1 failure) ... this is getting specific.
    
    For now, return NaN as it's not a simple pairwise average.
    """
    return float("nan")



# ---------------------------------------------------------------------------
# Ensemble wrappers (Smart usage of single vs group metrics)
# ---------------------------------------------------------------------------


def ensemble_metric(
    data: Union[List[Any], np.ndarray],
    metric_fn: Callable,
    **kwargs
) -> float:
    """
    Compute an ensemble/group metric over the given data.

    If R=2, computes the single-pair metric directly (optimization).
    If R>2, computes the average of the pairwise metrics (off-diagonal).

    Args:
        data: Either a list of items or a (4, R, R) counts stack.
        metric_fn: The single-pair metric function.
        **kwargs: Additional arguments passed to metric_fn.
    """
    # Detect R and Input Type
    is_stack = False
    if isinstance(data, np.ndarray) and data.ndim == 3 and data.shape[0] == 4:
        is_stack = True
        R = data.shape[1]
    elif isinstance(data, (list, tuple)):
        R = len(data)
    else:
        # Fallback to pairwise's validation or just assume it's a list if iterable
        try:
             R = len(data)
        except:
             raise ValueError("Data must be a list or counts stack.")

    if R < 2:
        return float("nan")

    if R == 2:
        # Single pair optimization
        if is_stack:
            return metric_fn(_counts_from_stack(data, 0, 1), **kwargs)
        else:
            return metric_fn(data[0], data[1], **kwargs)
    
    # Group case (Average pairwise)
    mat = pairwise(data, metric_fn, **kwargs)
    return _average_off_diagonal(mat)


def ensemble_pred_disagreement(preds_list: List[torch.Tensor]) -> float:
    """Wrapper for prediction disagreement using ensemble_metric."""
    return ensemble_metric(preds_list, pred_disagreement)


def ensemble_double_fault(preds_list: List[torch.Tensor], labels: torch.Tensor) -> float:
    """Wrapper for double fault using ensemble_metric."""
    # We need to construct counts if R=2 to use generic flow efficiently, 
    # OR we can just pass the preds_list to ensemble_metric with double_fault logic?
    # Actually double_fault takes (counts, N).
    # If we pass preds_list, we can't use double_fault directly because it expects counts.
    # So we should pass the stack to ensemble_metric.
    
    # For R=2 optimization, ensemble_metric handles stack->counts automatically.
    # But we need to build the stack first?
    # Building stack is expensive if we only need one pair.
    
    # Exception: if we want to optimize R=2 here specifically to avoid stack build:
    if len(preds_list) == 2:
        c_a = (preds_list[0] == labels).to(torch.float32)
        c_b = (preds_list[1] == labels).to(torch.float32)
        counts = compute_confusion_counts(c_a, c_b)
        return double_fault(counts, labels.numel())
        
    counts_stack = pairwise_confusion_counts(preds_list, labels)
    return ensemble_metric(counts_stack, double_fault, N=labels.numel())
