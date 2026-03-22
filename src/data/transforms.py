"""
Named transform presets.

Each preset is a (train_transform, eval_transform) pair.  Preset names are
versioned — if the meaning of a preset changes, create a new version
(e.g. cifar10_default_v2) rather than changing the existing one.

The preset name is included in ``dataset.transforms.preset`` and therefore
in cfg_hash, ensuring that different augmentation strategies produce
different hashes.
"""

from __future__ import annotations

from typing import Callable, Tuple

from torchvision import transforms

# ------------- CIFAR-10 presets ------------- #

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def _cifar10_default_v1():
    """Standard CIFAR-10 augmentation: RandomCrop + HFlip."""
    train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ]
    )
    eval_ = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ]
    )
    return train, eval_


def _cifar10_resizedcrop_v1():
    """RandomResizedCrop variant matching legacy main_ce.py behaviour."""
    train = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ]
    )
    eval_ = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ]
    )
    return train, eval_


# ------------- Registry ------------- #

_PRESETS: dict[str, Callable[[], Tuple]] = {
    "cifar10_default_v1": _cifar10_default_v1,
    "cifar10_resizedcrop_v1": _cifar10_resizedcrop_v1,
}


def get_transforms(preset: str) -> tuple:
    """
    Return ``(train_transform, eval_transform)`` for the given preset name.

    Raises ValueError if the preset is not registered.
    """
    factory = _PRESETS.get(preset)
    if factory is None:
        raise ValueError(
            f"Unknown transform preset '{preset}'. "
            f"Available: {sorted(_PRESETS.keys())}"
        )
    return factory()
