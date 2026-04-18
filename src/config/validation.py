"""Early-fail training config validation.

Called at the very start of training (before any data loading or model building)
to catch conflicting or orphaned config fields.  A field is "orphaned" when it
is only meaningful for a specific feature (e.g. ``lr_peak_epoch`` for the cyclic
scheduler) but that feature is inactive.

Raises ``ValueError`` with a human-readable list of every detected problem.
"""

from __future__ import annotations

from src.config.structured import ConTopoConfig


def validate_training_config(cfg: ConTopoConfig) -> None:
    """Fail early if the training config contains orphaned or conflicting fields.

    Args:
        cfg: Hydra DictConfig (top-level ConTopoConfig).

    Raises:
        ValueError: if one or more validation rules are violated.
    """
    errors: list[str] = []

    scheduler = cfg.training.scheduler
    lr_peak = cfg.training.lr_peak_epoch

    # cyclic scheduler requires lr_peak_epoch; lr_peak_epoch requires cyclic
    if scheduler == "cyclic" and lr_peak is None:
        errors.append(
            "scheduler=cyclic requires lr_peak_epoch to be set "
            "(e.g. training.lr_peak_epoch=2)"
        )
    if lr_peak is not None and scheduler != "cyclic":
        errors.append(
            f"lr_peak_epoch={lr_peak} is set but scheduler={scheduler!r} — "
            "lr_peak_epoch is only used by the cyclic scheduler; "
            "either set scheduler=cyclic or remove lr_peak_epoch"
        )

    # progressive resolution: min/max must be set together; ramps require min to be set
    prog_min = cfg.training.progressive_res_min
    prog_max = cfg.training.progressive_res_max
    ramp_start = cfg.training.progressive_res_start_ramp
    ramp_end = cfg.training.progressive_res_end_ramp
    prog_on = prog_min is not None

    if prog_on != (prog_max is not None):
        errors.append(
            "progressive_res_min and progressive_res_max must both be set or both be null"
        )
    if prog_on and ramp_start is None:
        errors.append(
            "progressive_res_min is set but progressive_res_start_ramp is null — "
            "set progressive_res_start_ramp (e.g. 0.75)"
        )
    if prog_on and ramp_end is None:
        errors.append(
            "progressive_res_min is set but progressive_res_end_ramp is null — "
            "set progressive_res_end_ramp (e.g. 1.0)"
        )
    if not prog_on and ramp_start is not None:
        errors.append(
            f"progressive_res_start_ramp={ramp_start} is set but progressive_res_min is null "
            "(orphaned field — only meaningful when progressive resolution is active)"
        )
    if not prog_on and ramp_end is not None:
        errors.append(
            f"progressive_res_end_ramp={ramp_end} is set but progressive_res_min is null "
            "(orphaned field — only meaningful when progressive resolution is active)"
        )
    if prog_on and prog_max is not None and prog_min >= prog_max:
        errors.append(
            f"progressive_res_min={prog_min} must be strictly less than "
            f"progressive_res_max={prog_max}"
        )

    # features that require loading_backend=ffcv
    backend = cfg.training.loading_backend
    if cfg.training.lr_tta and backend != "ffcv":
        errors.append(
            f"lr_tta=True requires loading_backend=ffcv (got {backend!r}) — "
            "TTA is only implemented in the FFCV training path"
        )
    if prog_on and backend != "ffcv":
        errors.append(
            f"progressive resolution requires loading_backend=ffcv (got {backend!r}) — "
            "set training.loading_backend=ffcv or disable progressive resolution"
        )

    # beton format fields are conditional on loading_backend=ffcv
    beton = cfg.training.beton
    beton_fields = {
        "beton.max_resolution": beton.max_resolution,
        "beton.jpeg_quality": beton.jpeg_quality,
        "beton.compress_probability": beton.compress_probability,
    }
    for name, val in beton_fields.items():
        if backend == "ffcv" and val is None:
            errors.append(f"loading_backend=ffcv requires training.{name} to be set")
        if backend != "ffcv" and val is not None:
            errors.append(
                f"training.{name}={val} is set but loading_backend={backend!r} "
                "(orphaned field — only meaningful when loading_backend=ffcv)"
            )

    if errors:
        raise ValueError(
            "Training config validation failed:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
