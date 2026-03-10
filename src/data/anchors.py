"""
Deterministic anchor / subset selection derived from dataset manifest.

Anchors are NOT model-dependent — they are derived solely from:
  - DatasetManifest (example_ids, labels, indices)
  - Anchor spec (per_class, strategy, order_by)

The anchor spec fields are: source_split, per_class, order_by, strategy.
These must be logged and included in behavior_input_hash when anchors
influence downstream behavior.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict

import torch
from torchvision import datasets

from src.data.manifest import DatasetManifest


def anchor_spec_hash(spec: Dict[str, Any]) -> str:
    """Deterministic hash of the anchor specification."""
    canonical = json.dumps(spec, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def select_anchors_from_manifest(
    manifest: DatasetManifest,
    per_class: int = 100,
    strategy: str = "per_class_first_n",
    order_by: str = "example_id",
    num_classes: int = 10,
) -> Dict[str, Any]:
    """
    Select anchor indices from a dataset manifest.

    Args:
        manifest: The DatasetManifest to select from.
        per_class: Number of anchors per class.
        strategy: Only ``per_class_first_n`` is currently supported.
        order_by: Ordering key — ``example_id`` (lexicographic) or
                  ``original_index`` (numeric).
        num_classes: Number of classes.

    Returns:
        Dict with keys:
          - anchor_indices:   list[int]  — indices into the manifest
          - anchor_example_ids: list[str]
          - anchor_labels:    torch.Tensor
          - spec: dict of the specification used
    """
    if strategy != "per_class_first_n":
        raise NotImplementedError(f"Anchor strategy '{strategy}' not implemented.")

    # Build (sort_key, manifest_idx) pairs per class
    class_indices: dict[int, list[tuple]] = {c: [] for c in range(num_classes)}
    for i, label in enumerate(manifest.labels.tolist()):
        if order_by == "example_id":
            sort_key = manifest.example_ids[i]
        elif order_by == "original_index":
            sort_key = int(manifest.original_indices[i])
        else:
            raise ValueError(f"Unknown order_by: {order_by}")
        class_indices[label].append((sort_key, i))

    # Sort each class and pick first N
    selected_indices: list[int] = []
    for c in range(num_classes):
        sorted_items = sorted(class_indices[c], key=lambda x: x[0])
        if len(sorted_items) < per_class:
            raise RuntimeError(
                f"Class {c}: only {len(sorted_items)} examples, need {per_class}"
            )
        for _, manifest_idx in sorted_items[:per_class]:
            selected_indices.append(manifest_idx)

    spec = {
        "source_split": manifest.split,
        "per_class": per_class,
        "strategy": strategy,
        "order_by": order_by,
        "num_classes": num_classes,
    }

    return {
        "anchor_indices": selected_indices,
        "anchor_example_ids": [manifest.example_ids[i] for i in selected_indices],
        "anchor_labels": manifest.labels[selected_indices],
        "spec": spec,
        "spec_hash": anchor_spec_hash(spec),
    }


def save_anchors(anchors: Dict[str, Any], path: str) -> None:
    """Save anchors dict to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(anchors, path)


def load_anchors(path: str) -> Dict[str, Any]:
    """Load anchors dict from disk."""
    return torch.load(path, weights_only=False)


def get_or_create_anchors(
    manifest: DatasetManifest,
    per_class: int = 100,
    strategy: str = "per_class_first_n",
    order_by: str = "example_id",
    artifacts_root: str = "artifacts",
) -> Dict[str, Any]:
    """
    Get cached anchors or create them from manifest.
    """
    spec = {
        "source_split": manifest.split,
        "per_class": per_class,
        "strategy": strategy,
        "order_by": order_by,
    }
    spec_h = anchor_spec_hash(spec)
    anchor_path = os.path.join(
        artifacts_root, "anchors",
        manifest.dataset_name, manifest.split, spec_h, "anchors.pt"
    )

    if os.path.isfile(anchor_path):
        return load_anchors(anchor_path)

    anchors = select_anchors_from_manifest(
        manifest, per_class=per_class, strategy=strategy, order_by=order_by
    )
    save_anchors(anchors, anchor_path)
    return anchors


