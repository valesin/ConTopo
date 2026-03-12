"""
Dataset-level manifest for stable alignment across runs.

Each manifest captures:
  - example_id:      content-hash of raw image bytes (SHA-256 truncated)
  - original_index:  position in the canonical dataset ordering
  - label:           ground-truth class

The manifest is stored both:
  1) Locally at ``artifacts/dataset_manifests/<dataset>/<manifest_hash>/manifest.pt``
  2) As an MLflow run with ``kind=dataset_manifest``
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Optional

import torch
from torchvision import datasets


@dataclass
class DatasetManifest:
    """Container for a per-split manifest."""
    example_ids: list[str]          # hex content hashes
    original_indices: torch.Tensor  # int64
    labels: torch.Tensor            # int64
    dataset_name: str
    split: str                      # "train", "val", "test"

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "example_ids": self.example_ids,
            "original_indices": self.original_indices,
            "labels": self.labels,
            "dataset_name": self.dataset_name,
            "split": self.split,
        }, path)

    @classmethod
    def load(cls, path: str) -> "DatasetManifest":
        d = torch.load(path, weights_only=False)
        return cls(**d)

    @property
    def manifest_hash(self) -> str:
        """Deterministic hash of manifest content (all example_ids + labels)."""
        content = "|".join(self.example_ids) + "|" + self.split
        return hashlib.sha256(content.encode()).hexdigest()[:16]


def _content_hash(raw_bytes: bytes) -> str:
    """SHA-256 truncated to 16 hex chars."""
    return hashlib.sha256(raw_bytes).hexdigest()[:16]


def _build_cifar10_manifest(root: str, train: bool, split_name: str) -> DatasetManifest:
    """
    Build a manifest for a CIFAR-10 split.

    Uses the raw PIL image bytes for content hashing.
    """
    ds = datasets.CIFAR10(root=root, train=train, download=True, transform=None)
    ids: list[str] = []
    indices: list[int] = []
    labels: list[int] = []

    for idx in range(len(ds)):
        img, label = ds[idx]
        # img is a PIL Image; convert to bytes for hashing
        raw = img.tobytes()
        ids.append(_content_hash(raw))
        indices.append(idx)
        labels.append(int(label))

    return DatasetManifest(
        example_ids=ids,
        original_indices=torch.tensor(indices, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        dataset_name="cifar10",
        split=split_name,
    )


def _log_manifest_to_mlflow(manifest: DatasetManifest, manifest_path: str) -> None:
    """Create an MLflow run for this manifest (best-effort)."""
    try:
        import mlflow
        from src.mlflow_utils import dataset_manifest_tags

        m_hash = manifest.manifest_hash
        tags = dataset_manifest_tags(manifest.dataset_name, manifest.split, m_hash)

        with mlflow.start_run(
            run_name=f"manifest_{manifest.dataset_name}_{manifest.split}",
            tags=tags,
            nested=True,
        ):
            mlflow.log_params({
                "dataset": manifest.dataset_name,
                "split": manifest.split,
                "num_examples": len(manifest.example_ids),
                "manifest_hash": m_hash,
            })
            mlflow.log_artifact(manifest_path, artifact_path="manifest")
    except Exception:
        pass  # best-effort



def get_or_create_manifest(
    dataset_name: str,
    split: str,
    data_root: str = "./dataset",
    artifacts_root: str = "artifacts",
    log_to_mlflow: bool = False,
    val_per_class: int = 500,
) -> DatasetManifest:
    """
    Return the manifest for the given dataset/split, creating it if absent.

    Path convention: ``<artifacts_root>/dataset_manifests/<dataset>/<split>/manifest.pt``

    For `split="val"`, only the val subset (first ``val_per_class`` per class
    by original order) is included — matching the split logic in
    ``src.data.loaders._split_train_val_indices``.

    If ``log_to_mlflow=True``, also logs the manifest as an MLflow run with
    ``kind=dataset_manifest``.
    """
    manifest_dir = os.path.join(artifacts_root, "dataset_manifests", dataset_name, split)
    manifest_path = os.path.join(manifest_dir, "manifest.pt")

    if os.path.isfile(manifest_path):
        return DatasetManifest.load(manifest_path)

    if dataset_name != "cifar10":
        raise NotImplementedError(f"Manifest generation for '{dataset_name}' is not implemented.")

    if split == "val":
        # Build from the train set, then filter to only the val indices
        full_train = _build_cifar10_manifest(data_root, train=True, split_name="train")
        targets = full_train.labels.tolist()
        class_counts: dict[int, int] = {c: 0 for c in range(10)}
        val_mask: list[int] = []
        for idx, y in enumerate(targets):
            if class_counts[y] < val_per_class:
                val_mask.append(idx)
                class_counts[y] += 1
        manifest = DatasetManifest(
            example_ids=[full_train.example_ids[i] for i in val_mask],
            original_indices=full_train.original_indices[val_mask],
            labels=full_train.labels[val_mask],
            dataset_name="cifar10",
            split="val",
        )
    elif split == "train":
        # Build full train and exclude val indices
        full_train = _build_cifar10_manifest(data_root, train=True, split_name="train")
        targets = full_train.labels.tolist()
        class_counts = {c: 0 for c in range(10)}
        val_set: set[int] = set()
        for idx, y in enumerate(targets):
            if class_counts[y] < val_per_class:
                val_set.add(idx)
                class_counts[y] += 1
        train_mask = [i for i in range(len(targets)) if i not in val_set]
        manifest = DatasetManifest(
            example_ids=[full_train.example_ids[i] for i in train_mask],
            original_indices=full_train.original_indices[train_mask],
            labels=full_train.labels[train_mask],
            dataset_name="cifar10",
            split="train",
        )
    else:
        # test split
        manifest = _build_cifar10_manifest(data_root, train=False, split_name=split)

    manifest.save(manifest_path)

    if log_to_mlflow:
        _log_manifest_to_mlflow(manifest, manifest_path)

    return manifest
