"""
Ensemble accuracy utilities.
"""

from __future__ import annotations

from typing import Dict, List

import torch


def ensemble_accuracy(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Accuracy of ensemble predictions (argmax of probs) vs. labels."""
    preds = probs.argmax(dim=1)
    return float((preds == labels).float().mean().item())


def component_accuracies(
    logits_list: List[torch.Tensor],
    labels: torch.Tensor,
) -> Dict[str, float]:
    """Compute per-component and summary accuracies."""
    accs = []
    for logits in logits_list:
        preds = logits.argmax(dim=1)
        accs.append(float((preds == labels).float().mean().item()))
    return {
        "mean_acc": sum(accs) / len(accs) if accs else 0.0,
        "max_acc": max(accs) if accs else 0.0,
        "per_component": accs,
        "num_components": len(accs),
    }
