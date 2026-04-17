"""
Tests for profile computation configuration.

Validates that profiling config is accessible at the new paths
(cfg.profiling.*) and that the skip flag works correctly.
"""

from __future__ import annotations

import os

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from src.config.structured import register_configs

# Register structured configs before tests
register_configs()

_CONF_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "conf")
_CONF_DIR = os.path.abspath(_CONF_DIR)


@pytest.fixture()
def cfg() -> DictConfig:
    """Compose full config using Hydra."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=_CONF_DIR):
        cfg = compose(config_name="config")
    return cfg


class TestProfilingProfilesSkipFlag:
    """Test the skip flag for profile computation."""

    def test_profiling_profiles_skip_default_false(self, cfg):
        """Default config should not skip profiles."""
        assert cfg.profiling.profiles.skip is False

    def test_profiling_profiles_section_exists(self, cfg):
        """profiling.profiles section should exist in config."""
        assert "profiles" in cfg.profiling
        assert "skip" in cfg.profiling.profiles
        assert cfg.profiling.profiles.skip is False

    def test_profiling_profiles_metrics_present(self, cfg):
        """profiling.profiles.metrics should be a non-empty list."""
        assert "metrics" in cfg.profiling.profiles
        assert len(cfg.profiling.profiles.metrics) > 0

    def test_pipeline_key_present_and_has_steps(self, cfg):
        """pipeline config group must be present and expose a steps list."""
        assert "pipeline" in cfg
        assert "steps" in cfg.pipeline
        assert len(cfg.pipeline.steps) > 0
