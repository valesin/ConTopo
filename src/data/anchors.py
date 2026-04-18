"""
Deterministic anchor / subset selection from dataset labels.

Anchors are NOT model-dependent — they are derived solely from:
  - ground-truth labels (torch.Tensor)
  - AnchorSpec (source_split, per_class, strategy, order_by, num_classes)

The ``AnchorSpec`` dataclass is the single source of truth for anchor
configuration. Defaults live in Hydra structured configs
(``AnchorSelectionConfig`` / ``AnchorsConfig``); core functions here
require all values to be supplied explicitly so that missing config
causes a loud, immediate error.
"""

from __future__ import annotations

import os
from typing import cast

import torch

from src.config.hash import compute_anchor_spec_hash
from src.types import AnchorOutput, AnchorSpec


def get_anchor_spec_dict(
    source_split: str,
    per_class: int,
    strategy: str,
    order_by: str,
    num_classes: int,
) -> AnchorSpec:
    """Return the plain dictionary representation of an anchor specification."""
    return {
        "source_split": source_split,
        "per_class": per_class,
        "strategy": strategy,
        "order_by": order_by,
        "num_classes": num_classes,
    }


def select_anchors(
    labels: torch.Tensor,
    source_split: str,
    per_class: int,
    strategy: str,
    order_by: str,
    num_classes: int,
) -> AnchorOutput:
    """
    Select anchor indices from ground-truth labels.

    Args:
        labels: int64 tensor of ground-truth class labels.
        source_split: Which data split anchors are sourced from.
        per_class: Number of anchors per class.
        strategy: Anchor selection strategy.
        order_by: Ordering strategy.
        num_classes: Total number of classes.

    Returns:
        Dict with keys:
          - anchor_indices:   list[int]  — indices into the label tensor
          - anchor_labels:    torch.Tensor
          - spec: dict of the specification used
          - spec_hash: deterministic hash of the spec
    """
    if strategy != "per_class_first_n":
        raise NotImplementedError(f"Anchor strategy '{strategy}' not implemented.")

    # Build (sort_key, idx) pairs per class — sort_key is always the
    # sequential dataset index now that manifest hashes are gone.
    class_indices: dict[int, list[tuple]] = {c: [] for c in range(num_classes)}
    for i, label in enumerate(labels.tolist()):
        sort_key = i  # original_index ordering
        class_indices[label].append((sort_key, i))

    # Sort each class and pick first N
    selected_indices: list[int] = []
    for c in range(num_classes):
        sorted_items = sorted(class_indices[c], key=lambda x: x[0])
        if len(sorted_items) < per_class:
            raise RuntimeError(
                f"Class {c}: only {len(sorted_items)} examples, need {per_class}"
            )
        for _, idx in sorted_items[:per_class]:
            selected_indices.append(idx)

    spec_dict = get_anchor_spec_dict(
        source_split, per_class, strategy, order_by, num_classes
    )
    spec_hash = compute_anchor_spec_hash(
        source_split, per_class, strategy, order_by, num_classes
    )

    return {
        "anchor_indices": selected_indices,
        "anchor_labels": labels[selected_indices],
        "spec": spec_dict,
        "spec_hash": spec_hash,
    }


def save_anchors(anchors: AnchorOutput, path: str) -> None:
    """Save anchors dict to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(anchors, path)


def load_anchors(path: str) -> AnchorOutput:
    """Load anchors dict from disk."""
    return cast(AnchorOutput, torch.load(path, weights_only=False))


def get_or_create_anchors(
    labels: torch.Tensor,
    source_split: str,
    per_class: int,
    strategy: str,
    order_by: str,
    num_classes: int,
    artifacts_root: str,
    dataset_name: str = "cifar10",
) -> AnchorOutput:
    """
    Get cached anchors or create them from labels.

    Args:
        labels: int64 tensor of ground-truth class labels.
        source_split: Which data split anchors are sourced from.
        per_class: Number of anchors per class.
        strategy: Anchor selection strategy.
        order_by: Ordering strategy.
        num_classes: Total number of classes.
        artifacts_root: Root directory for cached artifacts.
        dataset_name: Name of the dataset (used for cache path).
    """
    spec_h = compute_anchor_spec_hash(
        source_split, per_class, strategy, order_by, num_classes
    )
    anchor_path = os.path.join(
        artifacts_root,
        "anchors",
        dataset_name,
        source_split,
        spec_h,
        "anchors.pt",
    )

    if os.path.isfile(anchor_path):
        return load_anchors(anchor_path)

    anchors = select_anchors(
        labels, source_split, per_class, strategy, order_by, num_classes
    )
    save_anchors(anchors, anchor_path)
    return anchors
