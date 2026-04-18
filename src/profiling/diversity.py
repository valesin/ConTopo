"""
Pairwise classifier diversity metrics.

Registry + dispatcher + lazy context for ensemble diversity analysis.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property, partial
from typing import Any, Callable, NamedTuple

import numpy as np
import torch

# ─────────── data structures ───────────


@dataclass
class AgreementCounts:
    n11: int
    n00: int
    n10: int
    n01: int

    @property
    def tot(self) -> int:
        return self.n11 + self.n00 + self.n10 + self.n01

    @property
    def a(self) -> float:
        return self.n11 / self.tot if self.tot else float("nan")

    @property
    def b(self) -> float:
        return self.n10 / self.tot if self.tot else float("nan")

    @property
    def c(self) -> float:
        return self.n01 / self.tot if self.tot else float("nan")

    @property
    def d(self) -> float:
        return self.n00 / self.tot if self.tot else float("nan")


class Strategy(Enum):
    PAIRWISE_COUNTS = auto()
    PAIRWISE_LIST = auto()
    GLOBAL = auto()


class MetricSpec(NamedTuple):
    fn: Callable[..., float]
    strategy: Strategy
    context_key: str


# ─────────── lazy eval context ───────────


@dataclass
class EvalContext:
    preds: list[torch.Tensor]
    labels: torch.Tensor
    logits: list[torch.Tensor] | None = None

    @cached_property
    def counts_stack(self) -> np.ndarray:
        return _vectorized_counts(self.preds, self.labels)

    @property
    def num_classifiers(self) -> int:
        return len(self.preds)


# ─────────── registry ───────────

METRIC_REGISTRY: dict[str, MetricSpec] = {}


def register_metric(
    name: str,
    strategy: Strategy,
    context_key: str,
) -> Callable[[Callable[..., float]], Callable[..., float]]:
    def decorator(fn: Callable[..., float]) -> Callable[..., float]:
        METRIC_REGISTRY[name] = MetricSpec(fn, strategy, context_key)
        return fn

    return decorator


def compute_metrics(
    context: EvalContext,
    metrics: list[str],
    config: dict[str, Any] | None = None,
    reduce_group: bool = True,
) -> dict[str, float | np.ndarray]:
    results: dict[str, float | np.ndarray] = {}
    config = config or {}
    for name in metrics:
        spec = METRIC_REGISTRY[name]
        data = getattr(context, spec.context_key)
        fn_sig = inspect.signature(spec.fn)
        kwargs = {k: config[k] for k in fn_sig.parameters if k in config}
        if spec.strategy == Strategy.PAIRWISE_COUNTS:
            matrix = _apply_counts(data, partial(spec.fn, **kwargs))
            if reduce_group:
                results[name] = _avg_off_diag(matrix)
            else:
                results[name] = matrix
        elif spec.strategy == Strategy.PAIRWISE_LIST:
            matrix = _apply_list(data, spec.fn, **kwargs)
            if reduce_group:
                results[name] = _avg_off_diag(matrix)
            else:
                results[name] = matrix
        elif spec.strategy == Strategy.GLOBAL:
            results[name] = float(spec.fn(data, **kwargs))
        else:
            raise ValueError(f"Unknown strategy for {name}")
    return results


# ─────────── registered metrics ───────────


@register_metric("q_statistic", Strategy.PAIRWISE_COUNTS, "counts_stack")
def _q_statistic(c: AgreementCounts) -> float:
    ad, bc = c.n11 * c.n00, c.n10 * c.n01
    return float((ad - bc) / (ad + bc)) if (ad + bc) else np.nan


@register_metric("disagreement", Strategy.PAIRWISE_COUNTS, "counts_stack")
def _disagreement(c: AgreementCounts) -> float:
    return c.b + c.c


@register_metric("double_fault", Strategy.PAIRWISE_COUNTS, "counts_stack")
def _double_fault(c: AgreementCounts) -> float:
    return c.d


@register_metric("interrater_agreement", Strategy.PAIRWISE_COUNTS, "counts_stack")
def _interrater(c: AgreementCounts) -> float:
    num = 2 * (c.a * c.d - c.b * c.c)
    denom = (c.a + c.b) * (c.b + c.d) + (c.a + c.c) * (c.c + c.d)
    return num / denom if denom else np.nan


@register_metric("correlation", Strategy.PAIRWISE_COUNTS, "counts_stack")
def _correlation(c: AgreementCounts) -> float:
    num = c.a * c.d - c.b * c.c
    denom = np.sqrt((c.a + c.b) * (c.c + c.d) * (c.a + c.c) * (c.b + c.d))
    return num / denom if denom else np.nan


@register_metric("iou_top_n", Strategy.PAIRWISE_LIST, "logits")
def _iou_top_n(la: torch.Tensor, lb: torch.Tensor, n: int = 5) -> float:
    if la is lb:
        return 1.0
    ia = la.topk(n, dim=1).indices
    ib = lb.topk(n, dim=1).indices
    match = ia.unsqueeze(2) == ib.unsqueeze(1)
    inter = match.sum(dim=(1, 2)).float()
    union = 2 * n - inter
    return float((inter / union).mean().item())


# ─────────── internals ───────────


def _vectorized_counts(
    preds_list: list[torch.Tensor],
    labels: torch.Tensor,
) -> np.ndarray:
    R = len(preds_list)
    stack = torch.stack(preds_list).to(labels.device)
    C = (stack == labels.unsqueeze(0)).T.float()
    Cinv = 1.0 - C
    counts = np.zeros((4, R, R), dtype=int)
    counts[0] = (C.T @ C).cpu().numpy().astype(int)
    counts[1] = (Cinv.T @ Cinv).cpu().numpy().astype(int)
    counts[2] = (C.T @ Cinv).cpu().numpy().astype(int)
    counts[3] = (Cinv.T @ C).cpu().numpy().astype(int)
    return counts


def _apply_counts(
    stack: np.ndarray,
    fn: Callable[[AgreementCounts], float],
) -> np.ndarray:
    R = stack.shape[1]
    mat = np.full((R, R), np.nan)
    for i in range(R):
        for j in range(R):
            mat[i, j] = fn(
                AgreementCounts(
                    int(stack[0, i, j]),
                    int(stack[1, i, j]),
                    int(stack[2, i, j]),
                    int(stack[3, i, j]),
                )
            )
    return mat


def _apply_list(
    items: list[torch.Tensor],
    fn: Callable[..., float],
    **kw: Any,
) -> np.ndarray:
    R = len(items)
    mat = np.full((R, R), np.nan)
    for i in range(R):
        for j in range(R):
            mat[i, j] = fn(items[i], items[j], **kw)
    return mat


def _avg_off_diag(mat: np.ndarray) -> float:
    R = mat.shape[0]
    if R < 2:
        return float("nan")
    return float(np.nanmean(mat[~np.eye(R, dtype=bool)]))
