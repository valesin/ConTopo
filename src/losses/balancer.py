"""
GradNormBalancer — dynamic EMA-based loss balancing.

Extracted from the copy-pasted loop in main_ce.py / main_supcon.py / main_coscontr.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def grad_norm(loss: torch.Tensor, params: list[nn.Parameter]) -> torch.Tensor:
    """L2 norm of gradients of ``loss`` w.r.t. ``params`` (retain_graph)."""
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    flat = [g.detach().reshape(-1) for g in grads if g is not None]
    if not flat:
        return torch.tensor(0.0, device=loss.device)
    return torch.linalg.norm(torch.cat(flat), ord=2)


class GradNormBalancer:
    """
    Smoothed gradient-norm ratio scaling for a secondary loss term.

    Usage::

        balancer = GradNormBalancer(rho=cfg.loss.rho)
        for batch in loader:
            task_loss = ...
            topo_loss = ...
            scale = balancer.step(task_loss, topo_loss, measure_params)
            total_loss = task_loss + scale * topo_loss
    """

    def __init__(
        self,
        rho: float = 0.05,
        beta: float = 0.1,
        eps: float = 1e-8,
        lambda_max: float = 1e4,
    ):
        self.rho = rho
        self.beta = beta
        self.eps = eps
        self.lambda_max = lambda_max
        self._lambda_hat: torch.Tensor | None = None

    def step(
        self,
        task_loss: torch.Tensor,
        topo_loss: torch.Tensor,
        params: list[nn.Parameter],
    ) -> torch.Tensor:
        """Compute the current scaling coefficient for ``topo_loss``."""
        nt = grad_norm(task_loss, params)
        np_ = grad_norm(topo_loss, params)
        target = (self.rho * nt / (np_ + self.eps)).clamp(0.0, self.lambda_max).detach()
        if self._lambda_hat is None:
            self._lambda_hat = target
        else:
            self._lambda_hat = (1 - self.beta) * self._lambda_hat + self.beta * target
        return self._lambda_hat.detach()

    @property
    def lambda_hat(self) -> float:
        if self._lambda_hat is None:
            return 0.0
        return float(self._lambda_hat.detach().cpu())
