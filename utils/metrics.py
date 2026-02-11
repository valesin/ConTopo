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
from typing import Dict, List, Optional, Tuple, Union

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
) -> np.ndarray:
    """Apply a single-pair metric function over all pairs in a counts stack."""
    R = counts_stack.shape[1]
    result = np.full((R, R), np.nan, dtype=float)
    for i in range(R):
        for j in range(R):
            result[i, j] = fn(_counts_from_stack(counts_stack, i, j))
    return result


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
    R = len(preds_list)
    N = preds_list[0].numel()
    result = np.full((R, R), np.nan, dtype=float)
    for i in range(R):
        for j in range(R):
            diff = (preds_list[i] != preds_list[j]).to(torch.float32).sum().item()
            result[i, j] = diff / float(N)
    return result


def pairwise_norm_pred_disagreement(
    pred_disagreement: np.ndarray,
    ensemble_probs: Dict[Union[str, Tuple[int, int]], Dict[str, torch.Tensor]],
    labels: torch.Tensor,
) -> Dict[str, np.ndarray]:
    """
    Prediction disagreement normalised by ensemble error rate.

    For each ensemble method (soft, hard, max_confidence, conf_weighted),
    computes::

        norm_dis[i, j] = pred_disagreement[i, j] / (1 - ensemble_accuracy)

    Args:
        pred_disagreement: (R, R) matrix from ``pairwise_pred_disagreement``.
        ensemble_probs: dict from ``compute_all_ensemble_probs``, keyed by
            ``(i, j)`` tuples (i < j).  Each value maps method name to a
            probability tensor of shape [N, C].
        labels: 1-D ground-truth tensor (shape [N]).

    Returns:
        Dict mapping method name → (R, R) matrix.
    """
    R = pred_disagreement.shape[0]
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
                denom = 1.0 - ens_acc
                if denom == 0.0:
                    mat[i, j] = float("nan")
                else:
                    mat[i, j] = pred_disagreement[i, j] / denom
        result[method] = mat
    return result


def pairwise_double_fault(counts_stack: np.ndarray, N: int) -> np.ndarray:
    """Fraction of samples where both classifiers are wrong."""
    R = counts_stack.shape[1]
    result = np.full((R, R), np.nan, dtype=float)
    for i in range(R):
        for j in range(R):
            result[i, j] = counts_stack[1, i, j] / float(N)
    return result


def pairwise_output_correlation(probs_list: List[np.ndarray]) -> np.ndarray:
    """Pearson correlation over flattened probability vectors."""
    R = len(probs_list)
    result = np.full((R, R), np.nan, dtype=float)
    for i in range(R):
        for j in range(R):
            result[i, j] = _pearsonr_from_flat(
                probs_list[i].ravel(), probs_list[j].ravel()
            )
    return result


def pairwise_q_statistic(counts_stack: np.ndarray) -> np.ndarray:
    """Yule's Q statistic: (N11*N00 - N01*N10) / (N11*N00 + N01*N10)."""
    R = counts_stack.shape[1]
    result = np.full((R, R), np.nan, dtype=float)
    for i in range(R):
        for j in range(R):
            N11 = counts_stack[0, i, j]
            N00 = counts_stack[1, i, j]
            N01 = counts_stack[2, i, j]
            N10 = counts_stack[3, i, j]
            denom = N11 * N00 + N01 * N10
            if denom == 0:
                result[i, j] = float("nan")
            else:
                result[i, j] = float((N11 * N00 - N01 * N10) / denom)
    return result


def pairwise_param_cosine(param_vecs: List[np.ndarray]) -> np.ndarray:
    """Cosine similarity between flattened parameter vectors."""
    R = len(param_vecs)
    result = np.full((R, R), np.nan, dtype=float)
    for i in range(R):
        for j in range(R):
            a = param_vecs[i]
            b = param_vecs[j]
            if a is None or b is None:
                continue
            if hasattr(a, "detach"):
                a = a.detach().cpu().numpy()
            if hasattr(b, "detach"):
                b = b.detach().cpu().numpy()
            denom = float(norm(a) * norm(b))
            if denom == 0.0:
                continue
            result[i, j] = float(np.dot(a, b) / denom)
    return result
