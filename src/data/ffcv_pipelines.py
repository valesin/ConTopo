"""
FFCV augmentation pipeline builders.

Encapsulates all FFCV-specific image pipeline construction, keeping it out of
the loader and training script.  Import is guarded — this module is only usable
when the ``ffcv`` package is installed (optional dependency).

Two pipeline pairs are provided:

- ``build_ffcv_train_pipeline`` — RandomResizedCrop + RandomHFlip + fp16 output.
- ``build_ffcv_eval_pipeline``  — Resize + CenterCrop (standard eval).  When
  test-time augmentation (TTA) is enabled the caller is responsible for running
  two forward passes (original + HFlip) and averaging logits; this function
  returns only the single-pass pipeline.

Each builder returns ``(image_pipeline, label_pipeline)`` suitable for passing
directly to ``ffcv.loader.Loader`` as ``pipelines={"image": ..., "label": ...}``.

``build_ffcv_loader`` is a thin convenience wrapper around ``ffcv.loader.Loader``
that accepts the pipelines dict and common knobs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ffcv.loader import Loader as FfcvLoader


def build_ffcv_train_pipeline(
    image_size: int,
    device: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> tuple[list, list]:
    """Return (image_pipeline, label_pipeline) for FFCV training.

    Pipeline:
        RandomResizedCropRGBImageDecoder(output_size)
        → RandomHorizontalFlip(flip_prob=0.5)
        → ToTensor
        → ToDevice(device, non_blocking=True)
        → ToTorchImage
        → Convert(FLOAT16)
        → Normalize(mean, std)

    Args:
        image_size: Target crop size (square).
        device:     CUDA device index.
        mean:       Per-channel mean for normalisation.
        std:        Per-channel std for normalisation.

    Returns:
        (image_pipeline, label_pipeline)
    """
    import numpy as np
    import torch
    from ffcv.fields.decoders import RandomResizedCropRGBImageDecoder, IntDecoder
    from ffcv.transforms import (
        RandomHorizontalFlip,
        ToTensor,
        ToDevice,
        ToTorchImage,
        NormalizeImage,
    )

    mean_np = np.array(mean, dtype=np.float32) * 255
    std_np = np.array(std, dtype=np.float32) * 255

    image_pipeline = [
        RandomResizedCropRGBImageDecoder((image_size, image_size)),
        RandomHorizontalFlip(flip_prob=0.5),
        ToTensor(),
        ToDevice(torch.device(f"cuda:{device}"), non_blocking=True),
        ToTorchImage(),
        # NormalizeImage takes uint8 input and converts to the given output dtype in one step
        NormalizeImage(mean_np, std_np, np.dtype(np.float16)),
    ]
    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        ToDevice(torch.device(f"cuda:{device}"), non_blocking=True),
    ]
    return image_pipeline, label_pipeline


def build_ffcv_eval_pipeline(
    image_size: int,
    device: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    ratio: float = 256 / 224,
) -> tuple[list, list]:
    """Return (image_pipeline, label_pipeline) for FFCV evaluation.

    Pipeline:
        CenterCropRGBImageDecoder(output_size, ratio=ratio)
        → ToTensor
        → ToDevice(device, non_blocking=True)
        → ToTorchImage
        → Convert(FLOAT16)
        → Normalize(mean, std)

    For TTA: the caller runs two passes (original + HFlip) and averages logits.
    This function returns only the single-pass eval pipeline.

    Args:
        image_size: Target crop size (square).
        device:     CUDA device index.
        mean:       Per-channel mean for normalisation.
        std:        Per-channel std for normalisation.
        ratio:      Resize-before-crop ratio.  Default 256/224 matches the
                    standard ImageNet eval pre-processing.
    """
    import numpy as np
    import torch
    from ffcv.fields.decoders import CenterCropRGBImageDecoder, IntDecoder
    from ffcv.transforms import (
        ToTensor,
        ToDevice,
        ToTorchImage,
        NormalizeImage,
    )

    mean_np = np.array(mean, dtype=np.float32) * 255
    std_np = np.array(std, dtype=np.float32) * 255

    image_pipeline = [
        CenterCropRGBImageDecoder((image_size, image_size), ratio=ratio),
        ToTensor(),
        ToDevice(torch.device(f"cuda:{device}"), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(mean_np, std_np, np.dtype(np.float16)),
    ]
    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        ToDevice(torch.device(f"cuda:{device}"), non_blocking=True),
    ]
    return image_pipeline, label_pipeline


def build_ffcv_loader(
    beton_path: str,
    image_pipeline: list,
    label_pipeline: list,
    batch_size: int,
    num_workers: int,
    *,
    shuffled: bool = False,
    distributed: bool = False,
    seed: int | None = None,
) -> "FfcvLoader":
    """Construct and return an ``ffcv.loader.Loader``.

    Args:
        beton_path:     Path to the .beton dataset file.
        image_pipeline: Built by ``build_ffcv_train_pipeline`` or ``build_ffcv_eval_pipeline``.
        label_pipeline: Built by the same helpers.
        batch_size:     Mini-batch size.
        num_workers:    DataLoader worker processes.
        shuffled:       Use ``OrderOption.RANDOM`` (train) vs ``SEQUENTIAL`` (eval).
        distributed:    Use ``OrderOption.DISTRIBUTED`` for multi-GPU training.
        seed:           Optional RNG seed for shuffle reproducibility.
    """
    from ffcv.loader import Loader, OrderOption

    if distributed:
        order = OrderOption.DISTRIBUTED  # pyright: ignore[reportAttributeAccessIssue]
    elif shuffled:
        order = OrderOption.RANDOM
    else:
        order = OrderOption.SEQUENTIAL

    kwargs = {}
    if seed is not None:
        kwargs["seed"] = seed

    return Loader(
        beton_path,
        batch_size=batch_size,
        num_workers=num_workers,
        order=order,
        pipelines={"image": image_pipeline, "label": label_pipeline},
        **kwargs,
    )
