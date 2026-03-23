"""
Tests for Hydra configuration composition and structured configs.

Validates:
  - Config composes without errors under the new config topology
    (groups, profiling, analysis, execution replace pipeline)
  - New config sections are accessible at expected paths
  - EXCLUDED_KEYS correctly excludes non-training keys from cfg_hash
  - Adapter config is accessible at cfg.adapter.*
"""

from __future__ import annotations

import os

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from src.config.hash import cfg_hash, EXCLUDED_KEYS
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


class TestHydraComposition:
    """Verify Hydra config composes correctly with all groups."""

    def test_config_composes(self, cfg):
        assert isinstance(cfg, DictConfig)

    def test_model_group_present(self, cfg):
        assert "model" in cfg
        assert cfg.model.arch == "LinearResNet18"

    def test_loss_group_present(self, cfg):
        assert "loss" in cfg
        assert cfg.loss.name == "cross_entropy"

    def test_dataset_group_present(self, cfg):
        assert "dataset" in cfg
        assert cfg.dataset.name == "cifar10"

    def test_training_group_present(self, cfg):
        assert "training" in cfg
        assert cfg.training.epochs == 200

    def test_runtime_group_present(self, cfg):
        assert "runtime" in cfg

    def test_mlflow_group_present(self, cfg):
        assert "mlflow" in cfg
        assert cfg.mlflow.experiment_name == "contopo"

    def test_pipeline_removed(self, cfg):
        """pipeline must not exist — replaced by execution/profiling/analysis/groups."""
        assert "pipeline" not in cfg


class TestExecutionConfig:
    """Verify execution config group is accessible."""

    def test_execution_group_present(self, cfg):
        assert "execution" in cfg

    def test_execution_split_default(self, cfg):
        assert cfg.execution.split == "test"

    def test_execution_force_default(self, cfg):
        assert cfg.execution.force is False


class TestProfilingConfig:
    """Verify profiling config group is accessible."""

    def test_profiling_group_present(self, cfg):
        assert "profiling" in cfg

    def test_profiling_anchors_present(self, cfg):
        assert "anchors" in cfg.profiling
        assert cfg.profiling.anchors.per_class == 100
        assert cfg.profiling.anchors.source_split == "test"

    def test_profiling_profiles_present(self, cfg):
        assert "profiles" in cfg.profiling
        assert cfg.profiling.profiles.skip is False
        assert "cosine" in cfg.profiling.profiles.metrics

    def test_profiling_diagnostics_present(self, cfg):
        assert "diagnostics" in cfg.profiling
        assert cfg.profiling.diagnostics.morans_i is True


class TestAnalysisConfig:
    """Verify analysis config group is accessible."""

    def test_analysis_group_present(self, cfg):
        assert "analysis" in cfg

    def test_analysis_diversity_present(self, cfg):
        assert "diversity" in cfg.analysis
        assert cfg.analysis.diversity.enabled is True
        assert len(cfg.analysis.diversity.metrics) > 0

    def test_analysis_consistency_present(self, cfg):
        assert "consistency" in cfg.analysis
        assert cfg.analysis.consistency.enabled is True


class TestGroupsConfig:
    """Verify groups config group is accessible."""

    def test_groups_group_present(self, cfg):
        assert "groups" in cfg

    def test_groups_group_by(self, cfg):
        assert "topology" in cfg.groups.group_by
        assert "rho" in cfg.groups.group_by

    def test_groups_min_components(self, cfg):
        assert cfg.groups.min_components == 2


class TestEnsembleConfig:
    """Verify ensemble config contains only votes (discovery moved to groups)."""

    def test_ensemble_group_present(self, cfg):
        assert "ensemble" in cfg

    def test_ensemble_has_votes(self, cfg):
        assert "votes" in cfg.ensemble
        assert "soft" in cfg.ensemble.votes

    def test_ensemble_has_no_group_by(self, cfg):
        """group_by moved to cfg.groups — must not be in cfg.ensemble."""
        assert "group_by" not in cfg.ensemble

    def test_ensemble_has_no_min_components(self, cfg):
        """min_components moved to cfg.groups."""
        assert "min_components" not in cfg.ensemble


class TestAdapterConfig:
    """Verify adapter config is accessible via Hydra."""

    def test_adapter_group_present(self, cfg):
        assert "adapter" in cfg

    def test_adapter_has_epochs(self, cfg):
        assert cfg.adapter.epochs == 50

    def test_adapter_has_learning_rate(self, cfg):
        assert cfg.adapter.learning_rate == 0.001

    def test_adapter_has_batch_size(self, cfg):
        assert cfg.adapter.batch_size == 256

    def test_adapter_has_bias(self, cfg):
        assert cfg.adapter.bias is True

    def test_adapter_has_meta_type(self, cfg):
        assert cfg.adapter.meta_type == "meta_lr"

    def test_adapter_has_feature_type(self, cfg):
        assert cfg.adapter.feature_type == "logits"

    def test_adapter_has_similarity_metric(self, cfg):
        assert cfg.adapter.similarity_metric == "cosine"

    def test_adapter_has_init_seed(self, cfg):
        assert cfg.adapter.init_seed == 42

    def test_adapter_meta_split_has_no_strategy(self, cfg):
        """strategy field was removed from meta_split."""
        assert "strategy" not in cfg.adapter.meta_split

    def test_adapter_has_meta_split(self, cfg):
        ms = cfg.adapter.meta_split
        assert ms.seed == 42
        assert ms.fractions.train == 0.6
        assert ms.fractions.val == 0.2
        assert ms.fractions.holdout == 0.2


class TestExcludedKeys:
    """Verify EXCLUDED_KEYS includes all non-training config groups."""

    def test_groups_excluded(self):
        assert "groups" in EXCLUDED_KEYS

    def test_profiling_excluded(self):
        assert "profiling" in EXCLUDED_KEYS

    def test_analysis_excluded(self):
        assert "analysis" in EXCLUDED_KEYS

    def test_execution_excluded(self):
        assert "execution" in EXCLUDED_KEYS

    def test_ensemble_excluded(self):
        assert "ensemble" in EXCLUDED_KEYS

    def test_adapter_excluded(self):
        assert "adapter" in EXCLUDED_KEYS

    def test_migration_excluded(self):
        assert "migration" in EXCLUDED_KEYS

    def test_runtime_excluded(self):
        assert "runtime" in EXCLUDED_KEYS

    def test_mlflow_excluded(self):
        assert "mlflow" in EXCLUDED_KEYS

    def test_pipeline_not_in_excluded(self):
        """pipeline key no longer exists — not needed in EXCLUDED_KEYS."""
        assert "pipeline" not in EXCLUDED_KEYS

    def test_cfg_hash_ignores_execution(self, cfg):
        """Changing execution.split must NOT change model cfg_hash."""
        h1 = cfg_hash(cfg)
        cfg2 = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        cfg2.execution.split = "val"
        h2 = cfg_hash(cfg2)
        assert h1 == h2

    def test_cfg_hash_ignores_profiling(self, cfg):
        """Changing profiling.anchors must NOT change model cfg_hash."""
        h1 = cfg_hash(cfg)
        cfg2 = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        cfg2.profiling.anchors.per_class = 500
        h2 = cfg_hash(cfg2)
        assert h1 == h2

    def test_cfg_hash_ignores_groups(self, cfg):
        """Changing groups config must NOT change model cfg_hash."""
        h1 = cfg_hash(cfg)
        cfg2 = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        cfg2.groups.min_components = 10
        h2 = cfg_hash(cfg2)
        assert h1 == h2

    def test_cfg_hash_ignores_adapter(self, cfg):
        """Changing adapter config must NOT change model cfg_hash."""
        h1 = cfg_hash(cfg)
        cfg2 = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        cfg2.adapter.epochs = 999
        h2 = cfg_hash(cfg2)
        assert h1 == h2

    def test_cfg_hash_sensitive_to_loss(self, cfg):
        """Changing loss params MUST change model cfg_hash."""
        h1 = cfg_hash(cfg)
        cfg2 = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        cfg2.loss.rho = 99.9
        h2 = cfg_hash(cfg2)
        assert h1 != h2
