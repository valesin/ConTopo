"""
Deterministic anchor / subset selection derived from dataset manifest.

Anchors are NOT model-dependent — they are derived solely from:
  - DatasetManifest (example_ids, labels, indices)
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
from torchvision import datasets

from src.data.manifest import DatasetManifest


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


def select_anchors_from_manifest(
    manifest: DatasetManifest,
    spec: AnchorSpec,
) -> Dict[str, Any]:
    """
    Select anchor indices from a dataset manifest.

    Args:
        manifest: The DatasetManifest to select from.
        spec: Fully-populated ``AnchorSpec`` — no hidden defaults.

    Returns:
        Dict with keys:
          - anchor_indices:   list[int]  — indices into the manifest
          - anchor_example_ids: list[str]
          - anchor_labels:    torch.Tensor
          - spec: dict of the specification used
          - spec_hash: deterministic hash of the spec
    """
    if spec.strategy != "per_class_first_n":
        raise NotImplementedError(f"Anchor strategy '{spec.strategy}' not implemented.")

    # Build (sort_key, manifest_idx) pairs per class
    class_indices: dict[int, list[tuple]] = {c: [] for c in range(spec.num_classes)}
    for i, label in enumerate(manifest.labels.tolist()):
        if spec.order_by == "example_id":
            sort_key = manifest.example_ids[i]
        elif spec.order_by == "original_index":
            sort_key = int(manifest.original_indices[i])
        else:
            raise ValueError(f"Unknown order_by: {spec.order_by}")
        class_indices[label].append((sort_key, i))

    # Sort each class and pick first N
    selected_indices: list[int] = []
    for c in range(spec.num_classes):
        sorted_items = sorted(class_indices[c], key=lambda x: x[0])
        if len(sorted_items) < spec.per_class:
            raise RuntimeError(
                f"Class {c}: only {len(sorted_items)} examples, need {spec.per_class}"
            )
        for _, manifest_idx in sorted_items[:spec.per_class]:
            selected_indices.append(manifest_idx)

    return {
        "anchor_indices": selected_indices,
        "anchor_example_ids": [manifest.example_ids[i] for i in selected_indices],
        "anchor_labels": manifest.labels[selected_indices],
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
    manifest: DatasetManifest,
    spec: AnchorSpec,
    artifacts_root: str,
) -> Dict[str, Any]:
    """
    Get cached anchors or create them from manifest.

    Args:
        manifest: The DatasetManifest to select from.
        spec: Fully-populated ``AnchorSpec`` — no hidden defaults.
        artifacts_root: Root directory for cached artifacts.
    """
    spec_h = spec.hash
    anchor_path = os.path.join(
        artifacts_root, "anchors",
        manifest.dataset_name, manifest.split, spec_h, "anchors.pt"
    )

    if os.path.isfile(anchor_path):
        return load_anchors(anchor_path)

    anchors = select_anchors_from_manifest(manifest, spec)
    save_anchors(anchors, anchor_path)
    return anchors


