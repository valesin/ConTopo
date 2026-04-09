"""
Tests for cfg_hash stability, key ordering invariance, and
proper exclusion of runtime/mlflow/storage/hydra/groups/profiling/analysis/execution keys.
"""

import pytest
from omegaconf import OmegaConf

from src.config.hash import cfg_hash


def _make_cfg(**overrides):
    """Build a minimal DictConfig that mirrors the new two-tier layout."""
    base = {
        "schema_version": 1,
        "trial": 0,
        "seed": 100,
        "model": {
            "arch": "LinearResNet18",
            "embedding_dim": 256,
            "head": {"bias": True},
        },
        "loss": {
            "type": "cross_entropy",
            "topography_type": "ws",
            "topology": "torus",
            "rho": 0.05,
            "neighbourhood": {"type": "moore", "radius": 1},
        },
        "dataset": {
            "name": "cifar10",
            "split": {"strategy": "seeded_per_class", "seed": 0, "val_per_class": 500},
            "transforms": {"preset": "cifar10_resizedcrop_v1"},
        },
        "training": {
            "epochs": 200,
            "batch_size": 512,
            "learning_rate": 0.002,
            "optimiser": "adam",
            "weight_decay": 0.0,
            "momentum": 0.9,
            "scheduler": "none",
            "amp": False,
        },
    }
    base.update(overrides)
    return OmegaConf.create(base)


class TestCfgHash:
    def test_deterministic(self):
        c = _make_cfg()
        assert cfg_hash(c) == cfg_hash(c)

    def test_order_invariant(self):
        """Different key insertion order → same hash."""
        c1 = OmegaConf.create({"a": 1, "b": 2, "c": 3})
        c2 = OmegaConf.create({"c": 3, "a": 1, "b": 2})
        assert cfg_hash(c1) == cfg_hash(c2)

    def test_nested_order_invariant(self):
        c1 = OmegaConf.create({"x": {"b": 2, "a": 1}})
        c2 = OmegaConf.create({"x": {"a": 1, "b": 2}})
        assert cfg_hash(c1) == cfg_hash(c2)

    def test_different_values_different_hash(self):
        c1 = _make_cfg()
        overrides = {"loss": dict(_make_cfg().loss)}
        overrides["loss"]["rho"] = 0.1
        c2 = _make_cfg(**overrides)
        assert cfg_hash(c1) != cfg_hash(c2)

    def test_hash_is_16_hex(self):
        h = cfg_hash(_make_cfg())
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_float_precision(self):
        """Same float, different expressions → same hash."""
        c1 = OmegaConf.create({"rho": 0.05})
        c2 = OmegaConf.create({"rho": 5e-2})
        assert cfg_hash(c1) == cfg_hash(c2)

    def test_type_matters(self):
        """int 1 vs float 1.0 → different hash."""
        c1 = OmegaConf.create({"v": 1})
        c2 = OmegaConf.create({"v": 1.0})
        assert cfg_hash(c1) != cfg_hash(c2)

    # ── Exclusion tests ──

    def test_runtime_excluded(self):
        """Adding/changing runtime keys should NOT change hash."""
        c1 = _make_cfg()
        c2 = OmegaConf.merge(
            _make_cfg(),
            OmegaConf.create({"runtime": {"device": "cpu", "num_workers": 8}}),
        )
        assert cfg_hash(c1) == cfg_hash(c2)

    def test_mlflow_excluded(self):
        """mlflow keys should NOT change hash."""
        c1 = _make_cfg()
        c2 = OmegaConf.merge(
            _make_cfg(),
            OmegaConf.create({"mlflow": {"tracking_uri": "http://remote:5000"}}),
        )
        assert cfg_hash(c1) == cfg_hash(c2)

    def test_storage_excluded(self):
        c1 = _make_cfg()
        c2 = OmegaConf.merge(
            _make_cfg(),
            OmegaConf.create({"storage": {"backend": "zarr"}}),
        )
        assert cfg_hash(c1) == cfg_hash(c2)

    def test_hydra_excluded(self):
        c1 = _make_cfg()
        c2 = OmegaConf.merge(
            _make_cfg(),
            OmegaConf.create({"hydra": {"run": {"dir": "outputs/custom"}}}),
        )
        assert cfg_hash(c1) == cfg_hash(c2)

    def test_groups_excluded(self):
        """groups (discovery controls) should NOT change training cfg_hash."""
        c1 = _make_cfg()
        c2 = OmegaConf.merge(
            _make_cfg(),
            OmegaConf.create(
                {"groups": {"group_by": ["topology"], "min_components": 5}}
            ),
        )
        assert cfg_hash(c1) == cfg_hash(c2)

    def test_profiling_excluded(self):
        """profiling (anchor/profile params) should NOT change training cfg_hash."""
        c1 = _make_cfg()
        c2 = OmegaConf.merge(
            _make_cfg(),
            OmegaConf.create({"profiling": {"anchors": {"per_class": 200}}}),
        )
        assert cfg_hash(c1) == cfg_hash(c2)

    def test_analysis_excluded(self):
        """analysis (diversity/consistency config) should NOT change training cfg_hash."""
        c1 = _make_cfg()
        c2 = OmegaConf.merge(
            _make_cfg(),
            OmegaConf.create({"analysis": {"diversity": {"enabled": False}}}),
        )
        assert cfg_hash(c1) == cfg_hash(c2)

    def test_execution_excluded(self):
        """execution (split/force) should NOT change training cfg_hash."""
        c1 = _make_cfg()
        c2 = OmegaConf.merge(
            _make_cfg(),
            OmegaConf.create({"execution": {"split": "val", "force": True}}),
        )
        assert cfg_hash(c1) == cfg_hash(c2)

    def test_all_excluded_keys_together(self):
        """All excluded groups present → same hash as base."""
        c1 = _make_cfg()
        c2 = OmegaConf.merge(
            _make_cfg(),
            OmegaConf.create(
                {
                    "runtime": {"device": "cuda:1"},
                    "mlflow": {"experiment_name": "other"},
                    "storage": {"backend": "zarr"},
                    "hydra": {"verbose": True},
                    "groups": {"min_components": 10},
                    "profiling": {"anchors": {"per_class": 50}},
                    "analysis": {"consistency": {"enabled": False}},
                    "execution": {"split": "val", "force": True},
                }
            ),
        )
        assert cfg_hash(c1) == cfg_hash(c2)

    def test_topology_affects_hash(self):
        """loss.topology=torus vs grid → different hash."""
        c1 = _make_cfg()
        c2_loss = dict(_make_cfg().loss)
        c2_loss["topology"] = "grid"
        c2 = _make_cfg(**{"loss": c2_loss})
        assert cfg_hash(c1) != cfg_hash(c2)
