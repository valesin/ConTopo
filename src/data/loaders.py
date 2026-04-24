"""
Dataset-agnostic data loaders with deterministic train/val split and named transform presets.

Adding a new dataset
--------------------
1. Add a factory function ``_<name>_factory(root, train, transform, download=False)``
   that returns a torchvision-style Dataset with a ``.targets`` attribute.
2. Register it in ``_DATASET_FACTORIES``.
3. Add its class count to ``DATASET_NUM_CLASSES``.
4. Create ``conf/dataset/<name>.yaml`` with the required fields.
"""

import contextlib
import io
import os
from typing import Any, Callable

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets
from omegaconf import DictConfig

from src.data.transforms import get_transforms

# ─────────── per-dataset factories ───────────


def _cifar10_factory(
    root: str,
    train: bool,
    transform: Callable[[Any], Any] | None,
    download: bool = True,
) -> datasets.CIFAR10:
    """Wrapper around CIFAR10 that suppresses the 'Files already downloaded' stdout noise."""
    with contextlib.redirect_stdout(io.StringIO()):
        return datasets.CIFAR10(
            root=root, train=train, transform=transform, download=download
        )


def _imagenet100_factory(
    root: str,
    train: bool,
    transform: Callable[[Any], Any] | None,
    download: bool = False,
) -> datasets.ImageFolder:
    """ImageFolder loader for ImageNet100.

    Expects data at ``<root>/imagenet100/train/`` and ``<root>/imagenet100/val/``.
    The ``val/`` directory is the official 50-image/class held-out split used as the
    pipeline TEST split.  A custom validation split is carved from ``train/`` at
    runtime by ``_split_train_val_indices``.

    ``download`` is accepted but ignored — ImageNet100 must be present locally
    (typically via a symlink from the canonical data location).
    """
    subset = "train" if train else "val"
    path = os.path.join(root, "imagenet100", subset)
    return datasets.ImageFolder(root=path, transform=transform)


def _flowers102_factory(
    root: str,
    train: bool,
    transform: Callable[[Any], Any] | None,
    download: bool = True,
) -> Any:
    """Flowers102 loader adapting the split-string API to the factory's train-bool contract.

    Flowers102 has only 10 images/class in each official split.  To give
    ``_split_train_val_indices`` a usable pool, ``train=True`` concatenates the
    official 'train' and 'val' splits (20 images/class).  ``train=False`` returns
    the official 'test' split (~60 images/class on average).

    ``.targets`` is attached explicitly because Flowers102 exposes ``._labels``
    rather than the ``.targets`` attribute the rest of the pipeline expects.
    """
    if not train:
        ds = datasets.Flowers102(
            root=root, split="test", transform=transform, download=download
        )
        ds.targets = list(ds._labels)
        return ds
    ds_tr = datasets.Flowers102(
        root=root, split="train", transform=transform, download=download
    )
    ds_vl = datasets.Flowers102(
        root=root, split="val", transform=transform, download=download
    )
    combined = ConcatDataset([ds_tr, ds_vl])
    combined.targets = list(ds_tr._labels) + list(ds_vl._labels)
    return combined


# Registry: dataset name → factory(root, train, transform, download) → Dataset
_DATASET_FACTORIES: dict[str, Callable] = {
    "cifar10": _cifar10_factory,
    "imagenet100": _imagenet100_factory,
    "flowers102": _flowers102_factory,
}

DATASET_NUM_CLASSES: dict[str, int] = {
    "cifar10": 10,
    "imagenet100": 100,
    "flowers102": 102,
}


def get_num_classes(dataset_name: str) -> int:
    """Return the number of classes for a known dataset."""
    if dataset_name not in DATASET_NUM_CLASSES:
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. Add it to DATASET_NUM_CLASSES."
        )
    return DATASET_NUM_CLASSES[dataset_name]


def shutdown_dataloader_workers(loader: DataLoader | None) -> None:
    """Best-effort explicit DataLoader worker shutdown.

    Avoids occasional Python multiprocessing temp-dir cleanup race messages
    (e.g. /tmp/pymp-*) when scripts exit after using worker processes.
    """
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is None:
        return
    try:
        shutdown()
    except Exception:
        pass


def _split_train_val_indices(root: str, dataset_name: str, val_per_class: int = 500):
    """Deterministic train/val split from the dataset's training set.

    Picks the first ``val_per_class`` samples per class by the dataset's
    native ordering (alphabetical for ImageFolder, original order for CIFAR-*).

    Works for any dataset registered in ``_DATASET_FACTORIES`` whose training
    split exposes a ``.targets`` list of integer class labels.
    """
    factory = _DATASET_FACTORIES[dataset_name]
    base = factory(root=root, train=True, transform=None)
    targets = base.targets if hasattr(base, "targets") else list(base.train_labels)
    class_counts: dict[int, int] = {c: 0 for c in set(int(y) for y in targets)}
    val_idx: list[int] = []
    for idx, y in enumerate(targets):
        y_int = int(y)
        if class_counts[y_int] < val_per_class:
            val_idx.append(idx)
            class_counts[y_int] += 1
    all_idx = set(range(len(targets)))
    train_idx = sorted(all_idx - set(val_idx))
    val_idx = sorted(val_idx)
    return train_idx, val_idx


def get_dataset_loaders(cfg: DictConfig):
    """Build train / val / test loaders for any registered dataset.

    Dispatches on ``cfg.dataset.name`` via ``_DATASET_FACTORIES`` and on
    ``cfg.training.loading_backend`` to select torch vs FFCV.

    Returns:
        (train_loader, val_loader, test_loader)

        When ``loading_backend=ffcv`` **and** ``training.progressive_res_min`` is
        set, ``train_loader`` is a **list** of FFCV Loaders (one per discrete
        resolution step, ordered low-res → high-res).  In all other cases it is
        a single loader.  The training script uses ``_resolve_loader_for_epoch``
        to select the appropriate loader for each epoch.
    """
    backend = cfg.training.loading_backend
    if backend == "ffcv":
        return _get_ffcv_loaders(cfg)
    return _get_torch_loaders(cfg)


def _get_torch_loaders(cfg: DictConfig):
    """Torch DataLoader implementation (original path, unchanged)."""
    dataset_name = cfg.dataset.name
    factory = _DATASET_FACTORIES[dataset_name]

    preset = cfg.dataset.transforms.preset
    train_transform, eval_transform = get_transforms(preset)

    root = cfg.runtime.data_root
    val_per_class = cfg.dataset.split.val_per_class
    train_indices, val_indices = _split_train_val_indices(
        root, dataset_name, val_per_class
    )

    train_ds = factory(root=root, train=True, transform=train_transform)
    val_ds = factory(root=root, train=True, transform=eval_transform)
    test_ds = factory(root=root, train=False, transform=eval_transform)

    bs = int(cfg.training.batch_size)
    nw = int(cfg.runtime.num_workers)
    pin = (
        bool(cfg.runtime.pin_memory)
        if isinstance(cfg.runtime.pin_memory, str)
        else cfg.runtime.pin_memory
    )
    persistent = (
        bool(cfg.runtime.persistent_workers)
        if isinstance(cfg.runtime.persistent_workers, str)
        else cfg.runtime.persistent_workers
    )
    persistent = persistent and nw > 0

    train_loader = DataLoader(
        Subset(train_ds, train_indices),
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_indices),
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=persistent,
    )

    return train_loader, val_loader, test_loader


def _get_ffcv_loaders(cfg: DictConfig):
    """FFCV Loader implementation.

    Generates .beton files on demand (once per config) and builds FFCV Loaders.
    When ``training.progressive_res_min`` is set, returns a list of FFCV train
    loaders covering discrete resolution steps from min to max.
    """
    from src.data.beton_writer import get_or_write_beton
    from src.data.ffcv_pipelines import (
        build_ffcv_eval_pipeline,
        build_ffcv_loader,
        build_ffcv_train_pipeline,
    )

    dataset_name = cfg.dataset.name
    factory = _DATASET_FACTORIES[dataset_name]
    root = cfg.runtime.data_root
    val_per_class = cfg.dataset.split.val_per_class
    train_indices, val_indices = _split_train_val_indices(
        root, dataset_name, val_per_class
    )

    # Write beton files if absent
    train_beton = get_or_write_beton(cfg, "train", factory, train_indices)
    val_beton = get_or_write_beton(cfg, "val", factory, val_indices)
    test_beton = get_or_write_beton(cfg, "test", factory, None)

    bs = int(cfg.training.batch_size)
    nw = int(cfg.runtime.num_workers)

    # Device index: use 0 by default; resolved properly in the training script
    device_idx = 0

    mean = tuple(cfg.dataset.mean)
    std = tuple(cfg.dataset.std)

    # Val / test loaders (always single, at full image_size)
    image_size = int(cfg.dataset.image_size)
    # Cap ratio so CenterCropRGBImageDecoder never requests a region larger than
    # what is stored.  Images are stored at min(max_resolution, original_size);
    # for small datasets (e.g. CIFAR-10 32×32) they are stored at image_size,
    # so the ratio must be ≤ 1.0.
    max_stored = int(cfg.training.beton.max_resolution)
    effective_stored = min(max_stored, image_size)
    eval_ratio = min(256 / 224, effective_stored / image_size)
    val_img, val_lbl = build_ffcv_eval_pipeline(
        image_size, device_idx, mean, std, ratio=eval_ratio
    )
    test_img, test_lbl = build_ffcv_eval_pipeline(
        image_size, device_idx, mean, std, ratio=eval_ratio
    )
    val_loader = build_ffcv_loader(val_beton, val_img, val_lbl, bs, nw)
    test_loader = build_ffcv_loader(test_beton, test_img, test_lbl, bs, nw)

    # Train loader(s)
    prog_min = cfg.training.progressive_res_min
    prog_max = cfg.training.progressive_res_max

    if prog_min is not None and prog_max is not None:
        # Build one FFCV loader per discrete resolution step (low → high)
        resolutions = _progressive_resolutions(int(prog_min), int(prog_max))
        train_loaders = []
        for res in resolutions:
            img_pl, lbl_pl = build_ffcv_train_pipeline(res, device_idx, mean, std)
            train_loaders.append(
                build_ffcv_loader(train_beton, img_pl, lbl_pl, bs, nw, shuffled=True)
            )
        return train_loaders, val_loader, test_loader

    train_img, train_lbl = build_ffcv_train_pipeline(image_size, device_idx, mean, std)
    train_loader = build_ffcv_loader(
        train_beton, train_img, train_lbl, bs, nw, shuffled=True
    )
    return train_loader, val_loader, test_loader


def _progressive_resolutions(res_min: int, res_max: int, n_steps: int = 3) -> list[int]:
    """Return ``n_steps`` linearly-spaced integer resolutions from min to max (inclusive)."""
    if n_steps <= 1 or res_min == res_max:
        return [res_min]
    step = (res_max - res_min) / (n_steps - 1)
    return [int(round(res_min + i * step)) for i in range(n_steps)]


def get_cifar10_loaders(cfg: DictConfig):
    """Deprecated alias for ``get_dataset_loaders``. Use that instead."""
    return get_dataset_loaders(cfg)


def get_split_labels(cfg: DictConfig, split: str) -> "torch.Tensor":
    """Return ground-truth labels for a dataset split as an int64 tensor.

    Dispatches on ``cfg.dataset.name`` via ``_DATASET_FACTORIES``.
    """
    dataset_name = cfg.dataset.name
    factory = _DATASET_FACTORIES[dataset_name]
    root = cfg.runtime.data_root

    if split == "test":
        ds = factory(root=root, train=False, transform=None)
        targets = ds.targets if hasattr(ds, "targets") else list(ds.test_labels)
        return torch.tensor(targets, dtype=torch.long)

    val_per_class = cfg.dataset.split.val_per_class
    train_idx, val_idx = _split_train_val_indices(root, dataset_name, val_per_class)
    base = factory(root=root, train=True, transform=None)
    targets = base.targets if hasattr(base, "targets") else list(base.train_labels)

    if split == "val":
        return torch.tensor([targets[i] for i in val_idx], dtype=torch.long)
    elif split == "train":
        return torch.tensor([targets[i] for i in train_idx], dtype=torch.long)
    else:
        raise ValueError(f"Unknown split: {split}")


def get_dataset_eval_loader(
    cfg: DictConfig,
    split: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool | None = None,
) -> DataLoader:
    """Deterministic eval loader for the requested split of any registered dataset.

    Dispatches on ``cfg.dataset.name`` via ``_DATASET_FACTORIES``.
    Always uses eval (no-augmentation) transforms — intended for inference / profiling.

    Args:
        cfg: full Hydra config; reads dataset.name, dataset.transforms.preset,
             and dataset.split.val_per_class.
        split: one of "test", "val", "train".
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    dataset_name = cfg.dataset.name
    factory = _DATASET_FACTORIES[dataset_name]
    root = cfg.runtime.data_root
    preset = cfg.dataset.transforms.preset
    val_per_class = cfg.dataset.split.val_per_class

    _, eval_transform = get_transforms(preset)

    if split == "test":
        ds = factory(root=root, train=False, transform=eval_transform)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    if split in ("val", "train"):
        train_indices, val_indices = _split_train_val_indices(
            root, dataset_name, val_per_class
        )
        base_ds = factory(root=root, train=True, transform=eval_transform)
        indices = val_indices if split == "val" else train_indices
        return DataLoader(
            Subset(base_ds, indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    raise ValueError(f"Unknown split '{split}'. Expected 'test', 'val', or 'train'.")


def get_cifar10_eval_loader(
    root: str = "./dataset",
    batch_size: int = 256,
    num_workers: int = 2,
    pin_memory: bool | None = None,
    preset: str = "cifar10_resizedcrop_v1",
    split: str = "test",
    val_per_class: int = 500,
):
    """Deprecated. Use ``get_dataset_eval_loader`` instead.

    Kept for backward compatibility; wraps the generic loader using cifar10 factory
    directly so callers that pass individual kwargs still work.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    _, eval_transform = get_transforms(preset)

    if split == "test":
        ds = _cifar10_factory(root=root, train=False, transform=eval_transform)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    if split in ("val", "train"):
        train_indices, val_indices = _split_train_val_indices(
            root, "cifar10", val_per_class
        )
        base_ds = _cifar10_factory(root=root, train=True, transform=eval_transform)
        indices = val_indices if split == "val" else train_indices
        return DataLoader(
            Subset(base_ds, indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    raise ValueError(f"Unknown split '{split}'. Expected 'test', 'val', or 'train'.")
