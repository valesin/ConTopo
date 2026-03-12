"""
Tests for profile computation gating logic.

Validates the fix for the pipeline anti-pattern where Step 3
(03_compute_profiles.py) incorrectly gated profile generation on
``adapter.feature_type``.  Profiles are cross-config reusable artifacts
and must be generated regardless of the current adapter config.

See pipeline_best_practice.md for full rationale.
"""

from __future__ import annotations

import os
import sys

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from src.config.structured import register_configs

# Ensure the scripts directory is importable
_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "scripts")
sys.path.insert(0, os.path.abspath(_SCRIPTS_DIR))

# Register structured configs before tests
register_configs()

_CONF_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "conf")
_CONF_DIR = os.path.abspath(_CONF_DIR)


# Import the gating function from step 03
# We use importlib to handle the numeric module name
import importlib.util

_step03_path = os.path.join(
    os.path.dirname(__file__), os.pardir, "scripts", "03_compute_profiles.py"
)
_step03_spec = importlib.util.spec_from_file_location("step03", os.path.abspath(_step03_path))
_step03 = importlib.util.module_from_spec(_step03_spec)
_step03_spec.loader.exec_module(_step03)

_collect_profile_specs = _step03._collect_profile_specs


@pytest.fixture()
def cfg() -> DictConfig:
    """Compose full config using Hydra."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=_CONF_DIR):
        cfg = compose(config_name="config")
    return cfg


class TestProfileSpecsNotGatedOnFeatureType:
    """Profiles must be generated for ALL feature_type values."""

    def test_logits_still_generates_specs(self, cfg):
        """When feature_type=logits, profiles must still be generated."""
        cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg_copy.adapter.feature_type = "logits"
        specs = _collect_profile_specs(cfg_copy)
        assert len(specs) == 1, (
            "Profiles must be generated even when adapter.feature_type=logits. "
            "Step 5 may later run with feature_type=embeddings+profiles."
        )

    def test_embeddings_still_generates_specs(self, cfg):
        """When feature_type=embeddings, profiles must still be generated."""
        cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg_copy.adapter.feature_type = "embeddings"
        specs = _collect_profile_specs(cfg_copy)
        assert len(specs) == 1, (
            "Profiles must be generated even when adapter.feature_type=embeddings. "
            "Step 5 may later run with feature_type=embeddings+profiles."
        )

    def test_embeddings_plus_profiles_generates_specs(self, cfg):
        """When feature_type=embeddings+profiles, profiles must be generated."""
        cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg_copy.adapter.feature_type = "embeddings+profiles"
        specs = _collect_profile_specs(cfg_copy)
        assert len(specs) == 1

    def test_default_config_generates_specs(self, cfg):
        """Default config (feature_type=logits) must generate profile specs."""
        assert cfg.adapter.feature_type == "logits", "Sanity: default is logits"
        specs = _collect_profile_specs(cfg)
        assert len(specs) == 1, (
            "Default config must generate profile specs. "
            "This was the original bug: Step 3 skipped profiles when "
            "feature_type=logits, breaking later Step 5 runs with "
            "feature_type=embeddings+profiles."
        )


class TestProfileSpecContent:
    """Verify the spec dict has the right keys and values."""

    def test_spec_has_anchor_selection(self, cfg):
        specs = _collect_profile_specs(cfg)
        assert "anchor_selection" in specs[0]
        sel = specs[0]["anchor_selection"]
        assert isinstance(sel, dict)
        assert "per_class" in sel
        assert "strategy" in sel

    def test_spec_has_similarity_metric(self, cfg):
        specs = _collect_profile_specs(cfg)
        assert "similarity_metric" in specs[0]
        assert specs[0]["similarity_metric"] in ("cosine", "l2")

    def test_spec_reflects_adapter_config(self, cfg):
        """Spec must use the adapter config values."""
        specs = _collect_profile_specs(cfg)
        assert specs[0]["similarity_metric"] == cfg.adapter.similarity_metric
        assert specs[0]["anchor_selection"]["per_class"] == cfg.adapter.anchor_selection.per_class


class TestPipelineProfilesSkipFlag:
    """Test the explicit skip flag for profile computation."""

    def test_pipeline_profiles_skip_default_false(self, cfg):
        """Default config should not skip profiles."""
        skip = OmegaConf.select(cfg, "pipeline.profiles.skip", default=False)
        assert skip is False

    def test_pipeline_profiles_section_exists(self, cfg):
        """pipeline.profiles section should exist in config."""
        assert "profiles" in cfg.pipeline
        assert "skip" in cfg.pipeline.profiles
        assert cfg.pipeline.profiles.skip is False
