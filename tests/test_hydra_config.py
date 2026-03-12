"""
Tests for Hydra configuration composition and structured configs.

Validates:
  - Config composes without errors
  - Ensemble config is Hydra-managed (no yaml.safe_load needed)
  - EXCLUDED_KEYS correctly excludes non-training keys from cfg_hash
  - Structured config registration works
  - Adapter config is accessible at cfg.adapter.*
"""

from __future__ import annotations

import os

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, ListConfig, OmegaConf

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

    def test_pipeline_group_present(self, cfg):
        assert "pipeline" in cfg
        assert cfg.pipeline.split == "test"

    def test_mlflow_group_present(self, cfg):
        assert "mlflow" in cfg
        assert cfg.mlflow.experiment_name == "contopo"


class TestEnsembleInHydra:
    """Verify ensemble config is Hydra-managed (requirement #3)."""

    def test_ensemble_group_present(self, cfg):
        assert "ensemble" in cfg

    def test_ensemble_has_ensembles_list(self, cfg):
        ensembles = cfg.ensemble.ensembles
        assert isinstance(ensembles, (list, ListConfig))
        assert len(ensembles) > 0

    def test_ensemble_has_no_meta_split(self, cfg):
        """meta_split should NOT be in ensemble (moved to adapter)."""
        assert "meta_split" not in cfg.ensemble

    def test_ensemble_has_no_default_anchor_selection(self, cfg):
        """default_anchor_selection should NOT be in ensemble (moved to adapter)."""
        assert "default_anchor_selection" not in cfg.ensemble

    def test_ensemble_def_has_name(self, cfg):
        first = cfg.ensemble.ensembles[0]
        assert "name" in first

    def test_ensemble_def_has_selector(self, cfg):
        first = cfg.ensemble.ensembles[0]
        assert "selector" in first


class TestAdapterConfig:
    """Verify adapter config is accessible via Hydra (not yaml.safe_load)."""

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

    def test_adapter_has_dropout(self, cfg):
        assert isinstance(cfg.adapter.dropout, float)

    def test_adapter_has_meta_type(self, cfg):
        assert cfg.adapter.meta_type == "meta_lr"

    def test_adapter_has_feature_type(self, cfg):
        assert cfg.adapter.feature_type == "logits"

    def test_adapter_has_similarity_metric(self, cfg):
        assert cfg.adapter.similarity_metric == "cosine"

    def test_adapter_has_hidden_dim(self, cfg):
        assert cfg.adapter.hidden_dim == 128

    def test_adapter_has_init_seed(self, cfg):
        assert cfg.adapter.init_seed == 42

    def test_adapter_has_meta_split(self, cfg):
        assert "meta_split" in cfg.adapter
        ms = cfg.adapter.meta_split
        assert "seed" in ms
        assert "fractions" in ms
        assert ms.seed == 42
        assert ms.fractions.train == 0.6
        assert ms.fractions.val == 0.2
        assert ms.fractions.holdout == 0.2

    def test_adapter_has_anchor_selection(self, cfg):
        assert "anchor_selection" in cfg.adapter
        sel = cfg.adapter.anchor_selection
        assert sel.per_class == 100
        assert sel.strategy == "per_class_first_n"
        assert sel.order_by == "example_id"


class TestExcludedKeys:
    """Verify EXCLUDED_KEYS includes all non-training config groups."""

    def test_ensemble_excluded(self):
        assert "ensemble" in EXCLUDED_KEYS

    def test_adapter_excluded(self):
        assert "adapter" in EXCLUDED_KEYS

    def test_migration_excluded(self):
        assert "migration" in EXCLUDED_KEYS

    def test_pipeline_excluded(self):
        assert "pipeline" in EXCLUDED_KEYS

    def test_runtime_excluded(self):
        assert "runtime" in EXCLUDED_KEYS

    def test_mlflow_excluded(self):
        assert "mlflow" in EXCLUDED_KEYS

    def test_cfg_hash_ignores_ensemble(self, cfg):
        """Changing ensemble config must NOT change model cfg_hash."""
        h1 = cfg_hash(cfg)
        cfg2 = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg2.ensemble.ensembles = []
        h2 = cfg_hash(cfg2)
        assert h1 == h2

    def test_cfg_hash_ignores_adapter(self, cfg):
        """Changing adapter config must NOT change model cfg_hash."""
        h1 = cfg_hash(cfg)
        cfg2 = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg2.adapter.epochs = 999
        h2 = cfg_hash(cfg2)
        assert h1 == h2

    def test_cfg_hash_sensitive_to_loss(self, cfg):
        """Changing loss params MUST change model cfg_hash."""
        h1 = cfg_hash(cfg)
        cfg2 = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg2.loss.rho = 99.9
        h2 = cfg_hash(cfg2)
        assert h1 != h2
