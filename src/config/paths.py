"""
Centralized path resolution utilities.

All output paths are relative to ``cfg.runtime.outputs_root``.

Usage::

    from src.config.paths import get_anchors_dir

    anchors_dir = get_anchors_dir(cfg)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig


def get_outputs_root(cfg: DictConfig) -> Path:
    """Return the absolute path to the outputs root directory."""
    return Path(cfg.runtime.outputs_root).resolve()


def get_anchors_dir(cfg: DictConfig) -> Path:
    """Return the absolute path to the anchors directory."""
    return get_outputs_root(cfg) / cfg.runtime.paths.anchors


def get_mlflow_db_path(cfg: DictConfig) -> Path:
    """Return the absolute path to the MLflow SQLite database."""
    # Parse from tracking_uri which is like "sqlite:///outputs/mlflow.db"
    uri = cfg.mlflow.tracking_uri
    if uri.startswith("sqlite:///"):
        db_path = uri[len("sqlite:///") :]
        return Path(db_path).resolve()
    # For non-sqlite URIs, return None or raise
    return get_outputs_root(cfg) / "mlflow.db"


def ensure_output_dirs(cfg: DictConfig) -> None:
    """Create all output directories if they don't exist."""
    dirs = [
        get_outputs_root(cfg),
        get_anchors_dir(cfg),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
