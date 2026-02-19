"""
Pairwise classifier agreement and diversity metrics.

Refactored Implementation:
- Architecture: Registry + Dispatcher + Lazy Context
- Features: 
    - Automatic pairwise iteration (Vectorized or Generic).
    - Lazy computation of expensive intermediates (Confusion Stack).
    - Config-driven hyperparameter injection.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import cached_property, partial
from typing import Dict, List, Optional, Union, Any, Callable, NamedTuple

import numpy as np
import torch
from numpy.linalg import norm

# ---------------------------------------------------------------------------
# 1. Data Structures & Enums
# ---------------------------------------------------------------------------

@dataclass
class AgreementCounts:
    """Stores the four confusion matrix components for a pair of classifiers."""
    n11: int           # N11
    n00: int         # N00
    n10: int  # N10
    n01: int  # N01

    @property
    def tot(self) -> int:
        return (self.n11 + self.n00 + 
                self.n10 + self.n01)

    @property
    def a(self) -> float:
        """Proportion: n11 / tot"""
        return self.n11 / self.tot if self.tot > 0 else float('nan')

    @property
    def b(self) -> float:
        """Proportion: n10 / tot"""
        return self.n10 / self.tot if self.tot > 0 else float('nan')

    @property
    def c(self) -> float:
        """Proportion: n01 / tot"""
        return self.n01 / self.tot if self.tot > 0 else float('nan')

    @property
    def d(self) -> float:
        """Proportion: n00 / tot"""
        return self.n00 / self.tot if self.tot > 0 else float('nan')

class Strategy(Enum):
    """Defines how the metric should be computed over the ensemble."""
    PAIRWISE_COUNTS = auto()  # Iterates over pre-computed Confusion Matrix Stack
    PAIRWISE_LIST   = auto()  # Iterates over a raw list (preds, logits, weights) generic O(N^2)
    GLOBAL          = auto()  # Receives the entire object at once

class MetricSpec(NamedTuple):
    """Metadata for a registered metric."""
    fn: Callable
    strategy: Strategy
    context_key: str  # The attribute name in EvalContext to fetch data from

# ---------------------------------------------------------------------------
# 2. The Lazy Evaluation Context
# ---------------------------------------------------------------------------

@dataclass
class EvalContext:
    """
    Holds raw data and lazily computes derived structures (like confusion stacks).
    Acts as the dependency container for the dispatcher.
    """
    preds: List[torch.Tensor]
    labels: torch.Tensor
    
    # Optional inputs
    logits: Optional[List[torch.Tensor]] = None
    probs: Optional[List[np.ndarray]] = None
    params: Optional[List[np.ndarray]] = None # Weights/Gradient vectors

    @cached_property
    def counts_stack(self) -> np.ndarray:
        """
        Lazily computes the (4, R, R) confusion matrix stack using vectorized ops.
        Only runs if a metric requesting 'counts_stack' is called.
        """
        return _matrix_confusion_counts_vectorized(self.preds, self.labels)

    @property
    def num_classifiers(self) -> int:
        return len(self.preds)

# ---------------------------------------------------------------------------
# 3. Registry & Dispatcher
# ---------------------------------------------------------------------------

METRIC_REGISTRY: Dict[str, MetricSpec] = {}

def register_metric(name: str, strategy: Strategy, context_key: str):
    """Decorator to register a metric with its execution strategy."""
    def decorator(fn):
        METRIC_REGISTRY[name] = MetricSpec(fn, strategy, context_key)
        return fn
    return decorator

def compute_metrics(
    context: EvalContext,
    metrics: List[str],
    config: Dict[str, Any] = None,
    reduce_group: bool = True
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Main Entry Point.
    Computes requested metrics by injecting dependencies and handling iteration.
    """
    results = {}
    config = config or {}

    for name in metrics:
        if name not in METRIC_REGISTRY:
            raise ValueError(f"Metric '{name}' not found in registry.")
            
        spec = METRIC_REGISTRY[name]
        
        # 1. Fetch Data from Context
        if not hasattr(context, spec.context_key) or getattr(context, spec.context_key) is None:
             # Try fallback to raw dict if attribute is missing but key exists
             if spec.context_key in context.__dict__:
                 data = context.__dict__[spec.context_key]
             else:
                 raise ValueError(f"Metric '{name}' requires '{spec.context_key}', which is missing or None.")
        else:
            data = getattr(context, spec.context_key)

        # 2. Inject Hyperparameters from Config
        # Filter config to only include arguments accepted by the metric function
        fn_sig = inspect.signature(spec.fn)
        kwargs = {k: config[k] for k in fn_sig.parameters if k in config}

        # 3. Execute Strategy
        matrix = None
        if spec.strategy == Strategy.PAIRWISE_COUNTS:
            # Data is the (4, R, R) counts stack
            matrix = _apply_over_counts_stack(data, partial(spec.fn, **kwargs))
            
        elif spec.strategy == Strategy.PAIRWISE_LIST:
            # Data is a List[Tensor] or List[Array]
            matrix = _apply_over_generic_list(data, spec.fn, **kwargs)
            
        elif spec.strategy == Strategy.GLOBAL:
            # Data is passed directly (e.g., specific diversity measures)
            matrix = spec.fn(data, **kwargs)

        # 4. Format Output
        if reduce_group and spec.strategy != Strategy.GLOBAL:
            results[name] = _average_off_diagonal(matrix)
        else:
            results[name] = matrix

    return results

# ---------------------------------------------------------------------------
# 4. Metric Definitions (Math Kernels)
# ---------------------------------------------------------------------------
# Note: These functions define logic for ONE pair. The system handles the looping.

# --- Group A: Metrics using Confusion Counts ---

# As in Kuncheva 2003.
# Bounded between -1 and 1, where 
# 1 means perfect agreement, 
# 0 means random agreement, and 
# -1 means perfect disagreement.
# Symmetric and does not distinguish which classifier is which.
@register_metric("q_statistic", Strategy.PAIRWISE_COUNTS, "counts_stack")
def _calc_q_statistic(c: AgreementCounts) -> float:
    ad = c.n11 * c.n00
    bc = c.n10 * c.n01
    denom = ad + bc
    return float((ad - bc) / denom) if denom != 0 else np.nan

@register_metric("disagreement", Strategy.PAIRWISE_COUNTS, "counts_stack")
def _calc_disagreement(c: AgreementCounts) -> float:
    return c.b + c.c

@register_metric("double_fault", Strategy.PAIRWISE_COUNTS, "counts_stack")
def _calc_double_fault(c: AgreementCounts) -> float:
    return c.d

@register_metric("interrater_agreement", Strategy.PAIRWISE_COUNTS, "counts_stack")
def _calc_interrater_agreement(c: AgreementCounts) -> float:
    num = 2 * (c.a * c.d - c.b * c.c)
    denom = (c.a + c.b) * (c.b + c.d) + (c.a + c.c) * (c.c + c.d)
    return num / denom if denom != 0 else np.nan

@register_metric("correlation", Strategy.PAIRWISE_COUNTS, "counts_stack")
def _calc_correlation(c: AgreementCounts) -> float:    
    num = (c.a * c.d) - (c.b * c.c)
    denom = np.sqrt(
        (c.a + c.b) * (c.c + c.d) * (c.a + c.c) * (c.b + c.d)
    )
    return num / denom if denom != 0 else np.nan

# @register_metric("cohens_kappa", Strategy.PAIRWISE_COUNTS, "counts_stack")
# def _calc_cohens_kappa(c: AgreementCounts) -> float:
#     if c.tot == 0: return np.nan
#     pa = (c.n11 + c.n00) / c.tot
#     p_a_yes = (c.n11 + c.n10) / c.tot
#     p_b_yes = (c.n11 + c.n01) / c.tot
#     pe = p_a_yes * p_b_yes + (1 - p_a_yes) * (1 - p_b_yes)
#     return (pa - pe) / (1 - pe) if (1 - pe) != 0 else np.nan

# @register_metric("error_conditional_disagreement", Strategy.PAIRWISE_COUNTS, "counts_stack")
# def _calc_error_conditional_disagreement(c: AgreementCounts) -> float:
#     denom = c.n10 + c.n01 + c.n00
#     return (c.n10 + c.n01) / denom if denom > 0 else np.nan

# @register_metric("overall_agreement", Strategy.PAIRWISE_COUNTS, "counts_stack")
# def _calc_overall_agreement(c: AgreementCounts) -> float:
#     return (c.n11 + c.n00) / c.tot if c.tot > 0 else np.nan

# @register_metric("jaccard", Strategy.PAIRWISE_COUNTS, "counts_stack")
# def _calc_jaccard(c: AgreementCounts) -> float:
#     union = c.n11 + c.n10 + c.n01
#     return c.n11 / union if union > 0 else np.nan

# --- Group B: Metrics using Raw Lists ---

@register_metric("iou_top_n", Strategy.PAIRWISE_LIST, "logits")
def _calc_iou_top_n(logits_a: torch.Tensor, logits_b: torch.Tensor, n: int = 5) -> float:
    if logits_a is logits_b: return 1.0
    inds_a = logits_a.topk(n, dim=1).indices
    inds_b = logits_b.topk(n, dim=1).indices
    matches = (inds_a.unsqueeze(2) == inds_b.unsqueeze(1))
    inter = matches.sum(dim=(1, 2)).float()
    union = 2 * n - inter
    return (inter / union).mean().item()

# @register_metric("pred_disagreement", Strategy.PAIRWISE_LIST, "preds")
# def _calc_pred_disagreement(pred_a: torch.Tensor, pred_b: torch.Tensor) -> float:
#     N = pred_a.numel()
#     if N == 0: return np.nan
#     return (pred_a != pred_b).float().sum().item() / N

# @register_metric("output_correlation", Strategy.PAIRWISE_LIST, "probs")
# def _calc_output_correlation(probs_a: np.ndarray, probs_b: np.ndarray) -> float:
#     flat_a = probs_a.ravel().astype(np.float64)
#     flat_b = probs_b.ravel().astype(np.float64)
#     if len(flat_a) < 2: return np.nan
#     return float(np.corrcoef(flat_a, flat_b)[0, 1])

# @register_metric("param_cosine", Strategy.PAIRWISE_LIST, "params")
# def _calc_param_cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
#     if vec_a is None or vec_b is None: return np.nan
#     va = vec_a.detach().cpu().numpy() if hasattr(vec_a, 'detach') else vec_a
#     vb = vec_b.detach().cpu().numpy() if hasattr(vec_b, 'detach') else vec_b
    
#     denom = norm(va) * norm(vb)
#     return float(np.dot(va, vb) / denom) if denom > 0 else np.nan

# ---------------------------------------------------------------------------
# 5. Core Logic Helpers (Hidden)
# ---------------------------------------------------------------------------

def _matrix_confusion_counts_vectorized(preds_list: List[torch.Tensor], labels: torch.Tensor) -> np.ndarray:
    """Core vectorized logic for counts stack."""
    R = len(preds_list)
    N = labels.numel()
    
    preds_stack = torch.stack(preds_list).to(labels.device)
    C = (preds_stack == labels.unsqueeze(0)).T.float() # (N, R)
    
    n11_mat = C.T @ C
    C_inv = 1.0 - C
    n00_mat = C_inv.T @ C_inv
    n10_mat = C.T @ C_inv
    n01_mat = C_inv.T @ C
    
    counts_stack = np.zeros((4, R, R), dtype=int)
    counts_stack[0] = n11_mat.cpu().numpy().astype(int)
    counts_stack[1] = n00_mat.cpu().numpy().astype(int)
    counts_stack[2] = n10_mat.cpu().numpy().astype(int)
    counts_stack[3] = n01_mat.cpu().numpy().astype(int)
    return counts_stack

def _apply_over_counts_stack(counts_stack: np.ndarray, fn: Callable) -> np.ndarray:
    R = counts_stack.shape[1]
    result = np.full((R, R), np.nan, dtype=float)
    
    # Helper to extract struct
    def get_counts(i, j):
        return AgreementCounts(
            int(counts_stack[0, i, j]), int(counts_stack[1, i, j]),
            int(counts_stack[2, i, j]), int(counts_stack[3, i, j])
        )

    for i in range(R):
        for j in range(R):
            result[i, j] = fn(get_counts(i, j))
    return result

def _apply_over_generic_list(items: List[Any], fn: Callable, **kwargs) -> np.ndarray:
    R = len(items)
    result = np.full((R, R), np.nan, dtype=float)
    for i in range(R):
        for j in range(R):
            val = fn(items[i], items[j], **kwargs)
            if val is not None:
                result[i, j] = val
    return result

def _average_off_diagonal(mat: np.ndarray) -> float:
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]: return float("nan")
    R = mat.shape[0]
    if R < 2: return float("nan")
    return float(np.nanmean(mat[~np.eye(R, dtype=bool)]))