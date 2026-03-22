"""
cfg_hash — deterministic hash of experiment-semantic config only.

Excluded top-level keys (do NOT affect experiment results):
  runtime, mlflow, storage, hydra, pipeline

Included (experiment-semantic):
  schema_version, trial, seed, model.*, loss.*, dataset.*, training.*
"""

from __future__ import annotations

import hashlib
import json

from omegaconf import DictConfig, OmegaConf

EXCLUDED_KEYS = frozenset(
    {
        "runtime",
        "mlflow",
        "storage",
        "hydra",
        "pipeline",
        "ensemble",
        "adapter",
        "migration",
    }
)


def _deep_sort(obj):
    """Recursively sort dicts by key for canonical serialisation."""
    if isinstance(obj, dict):
        return {k: _deep_sort(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_deep_sort(v) for v in obj]
    return obj


def cfg_hash(cfg: DictConfig) -> str:
    """
    Deterministic SHA-256 (16 hex chars) of experiment-semantic config.

    Steps:
      1. Resolve interpolations via OmegaConf.
      2. Strip EXCLUDED_KEYS from the top level.
      3. Canonicalise via ``json.dumps(sort_keys=True)``.
      4. Return first 16 hex chars of SHA-256.
    """
    resolved = OmegaConf.to_container(cfg, resolve=True)
    for key in EXCLUDED_KEYS:
        resolved.pop(key, None)
    canonical = json.dumps(_deep_sort(resolved), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
