"""
Schema version and defaults application.

SCHEMA_VERSION is an integer that MUST be bumped whenever the meaning of
an existing config field changes (e.g. a transform preset is redefined).
It is included in cfg_hash to prevent hash collisions across schema versions.
"""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

SCHEMA_VERSION = 1


def apply_schema_defaults(cfg: DictConfig) -> DictConfig:
    """
    Ensure all required fields exist with their schema defaults.

    This is a safety net — Hydra group defaults should already supply
    most values; this handles edge cases where a new field is added
    but old YAML configs haven't been updated.
    """
    # Ensure schema_version present
    if OmegaConf.is_missing(cfg, "schema_version"):
        cfg.schema_version = SCHEMA_VERSION

    # Ensure seed derivation
    if cfg.get("seed") is None:
        cfg.seed = 100 + int(cfg.trial)

    return cfg
