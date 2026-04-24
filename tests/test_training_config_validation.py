from __future__ import annotations

import os

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from src.config.validation import validate_training_config


_CONF_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "conf")
_CONF_DIR = os.path.abspath(_CONF_DIR)


def _compose(overrides: list[str] | None = None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=_CONF_DIR):
        return compose(config_name="config", overrides=overrides or [])


def test_default_config_passes_validation() -> None:
    cfg = _compose()
    validate_training_config(cfg)


def test_topoloss_requires_all_topoloss_fields() -> None:
    cfg = _compose(["loss=topoloss", "loss.topoloss_scale=null"])
    with pytest.raises(ValueError, match="topography_type=topoloss requires"):
        validate_training_config(cfg)


def test_topoloss_fields_orphaned_when_topography_type_not_topoloss() -> None:
    cfg = _compose(["loss.topography_type=ws", "loss.topoloss_factor_h=8.0"])
    with pytest.raises(ValueError, match="orphaned field"):
        validate_training_config(cfg)


def test_ffcv_beton_fields_required_when_loading_backend_ffcv() -> None:
    cfg = _compose(
        ["training.loading_backend=ffcv", "training.beton.max_resolution=256"]
    )
    with pytest.raises(ValueError, match="loading_backend=ffcv requires"):
        validate_training_config(cfg)


def test_ffcv_beton_fields_orphaned_when_loading_backend_torch() -> None:
    cfg = _compose(
        ["training.loading_backend=torch", "training.beton.max_resolution=256"]
    )
    with pytest.raises(ValueError, match="orphaned field"):
        validate_training_config(cfg)
