"""
On-demand FFCV .beton file generator.

``get_or_write_beton`` transparently creates ``.beton`` files the first time they
are needed and returns the path on subsequent calls.  The path encodes all
parameters that affect file content, so the same config always maps to the same
file and it is safe to reuse across runs.

Path pattern::

    <beton.dir>/<dataset_name>_<split>_<max_res>px_q<quality>_j<compress_probability>.beton

Example::

    outputs/betons/imagenet100_train_500px_q90_j0.50.beton

The caller passes pre-computed split indices so that train/val splits are applied
before writing (only the chosen subset is stored in the beton file).
"""

from __future__ import annotations

import os
from typing import Callable

from omegaconf import DictConfig


def _beton_path(cfg: DictConfig, split: str) -> str:
    """Deterministic beton file path for (dataset, split, beton config)."""
    bc = cfg.training.beton  # format settings — hash-included
    beton_dir = cfg.runtime.beton.dir  # storage location — hash-excluded
    name = cfg.dataset.name
    res = bc.max_resolution
    quality = bc.jpeg_quality
    frac = f"{float(bc.compress_probability):.2f}"
    filename = f"{name}_{split}_{res}px_q{quality}_j{frac}.beton"
    return os.path.join(beton_dir, filename)


def get_or_write_beton(
    cfg: DictConfig,
    split: str,
    dataset_factory: Callable,
    indices: list[int] | None,
) -> str:
    """Return the .beton path for (dataset, split, beton config), writing it if absent.

    Args:
        cfg:             Full Hydra config. Uses ``cfg.training.beton`` (format
                         settings — hash-included), ``cfg.runtime.beton.dir``
                         (storage location — hash-excluded), and ``cfg.dataset``.
        split:           One of ``"train"``, ``"val"``, or ``"test"``.
        dataset_factory: Factory from ``_DATASET_FACTORIES`` — called as
                         ``factory(root, train=<bool>, transform=None)``.
        indices:         Pre-computed subset indices (from ``_split_train_val_indices``
                         for train/val, or ``None`` for the full test partition).

    Returns:
        Absolute path to the .beton file.
    """
    from ffcv.writer import DatasetWriter
    from ffcv.fields import RGBImageField, IntField
    from torch.utils.data import Subset

    path = _beton_path(cfg, split)

    if os.path.exists(path):
        return path

    os.makedirs(os.path.dirname(path), exist_ok=True)

    bc = cfg.training.beton  # format settings — hash-included
    data_root = cfg.runtime.data_root

    # For imagenet100-style datasets: train split = load train partition;
    # test/val also load train partition for the val split, and "val" partition
    # for the test split.
    is_train_source = split in ("train", "val")
    dataset = dataset_factory(root=data_root, train=is_train_source, transform=None)

    if indices is not None:
        dataset = Subset(dataset, indices)

    writer = DatasetWriter(
        path,
        {
            "image": RGBImageField(
                max_resolution=bc.max_resolution,
                jpeg_quality=bc.jpeg_quality,
                compress_probability=bc.compress_probability,
            ),
            "label": IntField(),
        },
    )
    writer.from_indexed_dataset(dataset)
    return path
