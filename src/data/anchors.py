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

import dataclasses
import hashlib
import json
import os
from typing import Any, Dict

import torch


@dataclasses.dataclass(frozen=True)
class AnchorSpec:
    """Immutable specification for anchor selection.

    All fields are required — no hidden defaults.  Defaults live in
    Hydra structured configs (``AnchorSelectionConfig``).
    """

    source_split: str
    per_class: int
    strategy: str
    order_by: str
    num_classes: int

    @property
    def hash(self) -> str:
        """Deterministic 16-char hex hash of this spec."""
        canonical = json.dumps(
            dataclasses.asdict(self), sort_keys=True, ensure_ascii=True
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary (e.g. for logging / serialisation)."""
        return dataclasses.asdict(self)


def anchor_spec_hash(spec: Dict[str, Any]) -> str:
    """Deterministic hash of an anchor specification dictionary.

    Kept for backward compatibility with code that already has a plain
    dict spec (e.g. cached artifacts).  New call-sites should use
    ``AnchorSpec.hash`` instead.
    """
    canonical = json.dumps(spec, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def select_anchors(
    labels: torch.Tensor,
    spec: AnchorSpec,
) -> Dict[str, Any]:
    """
    Select anchor indices from ground-truth labels.

    Args:
        labels: int64 tensor of ground-truth class labels.
        spec: Fully-populated ``AnchorSpec`` — no hidden defaults.

    Returns:
        Dict with keys:
          - anchor_indices:   list[int]  — indices into the label tensor
          - anchor_labels:    torch.Tensor
          - spec: dict of the specification used
          - spec_hash: deterministic hash of the spec
    """
    if spec.strategy != "per_class_first_n":
        raise NotImplementedError(f"Anchor strategy '{spec.strategy}' not implemented.")

    # Build (sort_key, idx) pairs per class — sort_key is always the
    # sequential dataset index now that manifest hashes are gone.
    class_indices: dict[int, list[tuple]] = {c: [] for c in range(spec.num_classes)}
    for i, label in enumerate(labels.tolist()):
        sort_key = i  # original_index ordering
        class_indices[label].append((sort_key, i))

    # Sort each class and pick first N
    selected_indices: list[int] = []
    for c in range(spec.num_classes):
        sorted_items = sorted(class_indices[c], key=lambda x: x[0])
        if len(sorted_items) < spec.per_class:
            raise RuntimeError(
                f"Class {c}: only {len(sorted_items)} examples, need {spec.per_class}"
            )
        for _, idx in sorted_items[: spec.per_class]:
            selected_indices.append(idx)

    return {
        "anchor_indices": selected_indices,
        "anchor_labels": labels[selected_indices],
        "spec": spec.to_dict(),
        "spec_hash": spec.hash,
    }


def save_anchors(anchors: Dict[str, Any], path: str) -> None:
    """Save anchors dict to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(anchors, path)


def load_anchors(path: str) -> Dict[str, Any]:
    """Load anchors dict from disk."""
    return torch.load(path, weights_only=False)


def get_or_create_anchors(
    labels: torch.Tensor,
    spec: AnchorSpec,
    artifacts_root: str,
    dataset_name: str = "cifar10",
) -> Dict[str, Any]:
    """
    Get cached anchors or create them from labels.

    Args:
        labels: int64 tensor of ground-truth class labels.
        spec: Fully-populated ``AnchorSpec`` — no hidden defaults.
        artifacts_root: Root directory for cached artifacts.
        dataset_name: Name of the dataset (used for cache path).
    """
    spec_h = spec.hash
    anchor_path = os.path.join(
        artifacts_root,
        "anchors",
        dataset_name,
        spec.source_split,
        spec_h,
        "anchors.pt",
    )

    if os.path.isfile(anchor_path):
        return load_anchors(anchor_path)

    anchors = select_anchors(labels, spec)
    save_anchors(anchors, anchor_path)
    return anchors
