"""
CIFAR-10 data loaders with deterministic train/val split and named transform presets.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from omegaconf import DictConfig

from src.data.transforms import get_transforms


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


def _split_train_val_indices(root: str, val_per_class: int = 500):
    """
    Deterministic 45k/5k split from the CIFAR-10 train set.
    Picks the first ``val_per_class`` samples per class by original order.
    """
    base = datasets.CIFAR10(root=root, train=True, transform=None, download=True)
    targets = base.targets if hasattr(base, "targets") else base.train_labels
    class_counts = {c: 0 for c in range(10)}
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


def get_cifar10_loaders(cfg: DictConfig):
    """
    Build train / val / test DataLoaders for CIFAR-10 using Hydra config.

    Uses named transform presets from ``cfg.dataset.transforms.preset``.
    Runtime knobs (num_workers, pin_memory, etc.) come from ``cfg.runtime``.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Get transforms from named preset
    preset = cfg.dataset.transforms.preset
    train_transform, eval_transform = get_transforms(preset)

    root = cfg.runtime.data_root
    val_per_class = cfg.dataset.split.val_per_class
    train_indices, val_indices = _split_train_val_indices(root, val_per_class)

    train_ds = datasets.CIFAR10(
        root=root, train=True, transform=train_transform, download=True
    )
    val_ds = datasets.CIFAR10(
        root=root, train=True, transform=eval_transform, download=True
    )
    test_ds = datasets.CIFAR10(
        root=root, train=False, transform=eval_transform, download=True
    )

    bs = cfg.training.batch_size
    nw = cfg.runtime.num_workers
    pin = cfg.runtime.pin_memory
    persistent = cfg.runtime.persistent_workers and nw > 0

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


def get_split_labels(cfg: DictConfig, split: str) -> "torch.Tensor":
    """Return ground-truth labels for a CIFAR-10 split as an int64 tensor.

    This is the lightweight replacement for the old manifest.labels look-up.
    """
    root = cfg.runtime.data_root
    if split == "test":
        ds = datasets.CIFAR10(root=root, train=False, download=True, transform=None)
        targets = ds.targets if hasattr(ds, "targets") else ds.test_labels
        return torch.tensor(targets, dtype=torch.long)

    val_per_class = cfg.dataset.split.val_per_class
    train_idx, val_idx = _split_train_val_indices(root, val_per_class)
    base = datasets.CIFAR10(root=root, train=True, download=True, transform=None)
    targets = base.targets if hasattr(base, "targets") else base.train_labels

    if split == "val":
        return torch.tensor([targets[i] for i in val_idx], dtype=torch.long)
    elif split == "train":
        return torch.tensor([targets[i] for i in train_idx], dtype=torch.long)
    else:
        raise ValueError(f"Unknown split: {split}")


def get_cifar10_eval_loader(
    root: str = "./dataset",
    batch_size: int = 256,
    num_workers: int = 2,
    pin_memory: bool | None = None,
    preset: str = "cifar10_resizedcrop_v1",
    split: str = "test",
    val_per_class: int = 500,
):
    """Deterministic eval loader for the requested split.

    Always uses eval (no-augmentation) transforms regardless of split, since
    this loader is intended for inference / profiling, not for training.

    Args:
        split: one of "test", "val", "train".
        val_per_class: controls the train/val boundary (must match training).
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    _, eval_transform = get_transforms(preset)

    if split == "test":
        ds = datasets.CIFAR10(
            root=root, train=False, download=True, transform=eval_transform
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    if split in ("val", "train"):
        train_indices, val_indices = _split_train_val_indices(root, val_per_class)
        base_ds = datasets.CIFAR10(
            root=root, train=True, download=True, transform=eval_transform
        )
        indices = val_indices if split == "val" else train_indices
        return DataLoader(
            Subset(base_ds, indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    raise ValueError(f"Unknown split '{split}'. Expected 'test', 'val', or 'train'.")
