"""
Metric Registry for Ensemble Diversity Analysis.

Provides a unified registry of diversity metrics with both **pairwise** (R×R matrix)
and **group** (scalar) computation variants.  Used by ``exp_diversity.py`` for all
diversity analyses (``--model``, ``--config``, ``--config --combinatorial``).

Registry layout
---------------
Each entry maps a human-readable config name to a ``(pairwise_fn, group_fn)`` pair:

    ``METRIC_REGISTRY[name] = (pw_callable, gp_callable)``

*  ``pw_callable(ctx)``  → ``np.ndarray`` of shape ``(R, R)`` with pairwise values.
*  ``gp_callable(ctx)``  → ``float`` scalar (typically the off-diagonal mean of the
   pairwise matrix, but may be a different group-level statistic).

Context dict
------------
Both callables receive a *context dict* ``ctx`` containing pre-computed data::

    {
        "preds_list":   List[Tensor],       # per-model argmax predictions  [N]
        "probs_list":   List[ndarray],      # per-model softmax probs       [N, C]
        "logits_list":  List[Tensor],       # per-model raw logits          [N, C]
        "counts_stack": ndarray(4, R, R),   # confusion count matrices
        "labels":       Tensor,             # ground-truth labels           [N]
        "N":            int,                # number of samples
    }

ALL_ENSEMBLE_METHODS lists the ensemble combination strategies supported by
``ensemble_probs_for_subset``: soft, hard, max_confidence, conf_weighted.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from utils.metrics import (
    pairwise_pred_disagreement,
    pairwise_q_statistic,
    pairwise_output_correlation,
    pairwise_double_fault,
    pairwise_jaccard,
    pairwise_cohens_kappa,
    pairwise_correctness_disagreement,
    pairwise_error_conditional_disagreement,
    pairwise_overall_agreement,
    pairwise_asym_ratio,
    pairwise_mcnemar_p,
    pairwise_iou_top_n,
    group_pred_disagreement,
    group_q_statistic,
    group_output_correlation,
    group_double_fault,
    group_jaccard,
    group_cohens_kappa,
    group_correctness_disagreement,
    group_error_conditional_disagreement,
    group_overall_agreement,
    group_asym_ratio,
    group_mcnemar_p,
    group_iou_top_n,
)


# ---------------------------------------------------------------------------
# Pairwise wrappers  (ctx → R×R ndarray)
# ---------------------------------------------------------------------------

def _pw_disagreement(ctx: dict) -> np.ndarray:
    return pairwise_pred_disagreement(ctx["preds_list"])

def _pw_q_statistic(ctx: dict) -> np.ndarray:
    return pairwise_q_statistic(ctx["counts_stack"])

def _pw_output_correlation(ctx: dict) -> np.ndarray:
    return pairwise_output_correlation(ctx["probs_list"])

def _pw_double_fault(ctx: dict) -> np.ndarray:
    return pairwise_double_fault(ctx["counts_stack"], ctx["N"])

def _pw_jaccard(ctx: dict) -> np.ndarray:
    return pairwise_jaccard(ctx["counts_stack"])

def _pw_cohens_kappa(ctx: dict) -> np.ndarray:
    return pairwise_cohens_kappa(ctx["counts_stack"])

def _pw_correctness_disagreement(ctx: dict) -> np.ndarray:
    return pairwise_correctness_disagreement(ctx["counts_stack"])

def _pw_error_conditional_disagreement(ctx: dict) -> np.ndarray:
    return pairwise_error_conditional_disagreement(ctx["counts_stack"])

def _pw_overall_agreement(ctx: dict) -> np.ndarray:
    return pairwise_overall_agreement(ctx["counts_stack"])

def _pw_asym_ratio(ctx: dict) -> np.ndarray:
    return pairwise_asym_ratio(ctx["counts_stack"])

def _pw_mcnemar_p(ctx: dict) -> np.ndarray:
    return pairwise_mcnemar_p(ctx["counts_stack"])

def _pw_iou_top_n(ctx: dict) -> np.ndarray:
    return pairwise_iou_top_n(ctx["logits_list"])


# ---------------------------------------------------------------------------
# Group wrappers  (ctx → float scalar)
# ---------------------------------------------------------------------------

def _gp_disagreement(ctx: dict) -> float:
    return group_pred_disagreement(ctx["preds_list"])

def _gp_q_statistic(ctx: dict) -> float:
    return group_q_statistic(ctx["counts_stack"])

def _gp_output_correlation(ctx: dict) -> float:
    return group_output_correlation(ctx["probs_list"])

def _gp_double_fault(ctx: dict) -> float:
    return group_double_fault(ctx["counts_stack"], ctx["N"])

def _gp_jaccard(ctx: dict) -> float:
    return group_jaccard(ctx["counts_stack"])

def _gp_cohens_kappa(ctx: dict) -> float:
    return group_cohens_kappa(ctx["counts_stack"])

def _gp_correctness_disagreement(ctx: dict) -> float:
    return group_correctness_disagreement(ctx["counts_stack"])

def _gp_error_conditional_disagreement(ctx: dict) -> float:
    return group_error_conditional_disagreement(ctx["counts_stack"])

def _gp_overall_agreement(ctx: dict) -> float:
    return group_overall_agreement(ctx["counts_stack"])

def _gp_asym_ratio(ctx: dict) -> float:
    return group_asym_ratio(ctx["counts_stack"])

def _gp_mcnemar_p(ctx: dict) -> float:
    return group_mcnemar_p(ctx["counts_stack"])

def _gp_iou_top_n(ctx: dict) -> float:
    return group_iou_top_n(ctx["logits_list"])


# ---------------------------------------------------------------------------
# Registry  —  config name → (pairwise_fn, group_fn)
# ---------------------------------------------------------------------------

METRIC_REGISTRY: Dict[str, Tuple[Callable, Callable]] = {
    "pred_disagreement":              (_pw_disagreement,                _gp_disagreement),
    "q_statistic":                    (_pw_q_statistic,                 _gp_q_statistic),
    "output_correlation":             (_pw_output_correlation,          _gp_output_correlation),
    "double_fault":                   (_pw_double_fault,                _gp_double_fault),
    "jaccard":                        (_pw_jaccard,                     _gp_jaccard),
    "cohens_kappa":                   (_pw_cohens_kappa,                _gp_cohens_kappa),
    "correctness_disagreement":       (_pw_correctness_disagreement,    _gp_correctness_disagreement),
    "error_conditional_disagreement": (_pw_error_conditional_disagreement, _gp_error_conditional_disagreement),
    "overall_agreement":              (_pw_overall_agreement,           _gp_overall_agreement),
    "asym_ratio":                     (_pw_asym_ratio,                  _gp_asym_ratio),
    "mcnemar_p":                      (_pw_mcnemar_p,                   _gp_mcnemar_p),
    "iou_top_n":                      (_pw_iou_top_n,                   _gp_iou_top_n),
}

# All ensemble methods supported by ensemble_probs_for_subset
ALL_ENSEMBLE_METHODS: List[str] = ["soft", "hard", "max_confidence", "conf_weighted"]
