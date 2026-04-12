from __future__ import annotations

import pytest

from src.config.hash import IDEMPOTENCY_REGISTRY, identity_hash
from src.mlflow_schema_logger import TELEMETRY_SCHEMA


def _base_fields(kind: str) -> dict[str, str]:
    if kind == "model":
        return {
            "schema_version": "1",
            "trial": "0",
            "seed": "42",
            "model.arch": "resnet18",
            "loss.rho": "0.0",
            "dataset.name": "cifar10",
            "training.epochs": "1",
        }
    if kind == "inference":
        return {"trained_model_run_id": "run_1", "split": "test"}
    if kind == "category_similarity_profile":
        return {
            "parent_run_id": "run_1",
            "anchor_spec_hash": "a123",
            "similarity_metric": "cosine",
            "split": "test",
        }
    if kind == "diagnostics":
        return {
            "parent_run_id": "run_1",
            "diagnostic_metric": "morans_i",
            "split": "test",
        }
    if kind == "ensemble":
        return {
            "component_set_hash": "cs123",
            "split": "test",
            "feature_type": "logits",
            "method": "soft",
        }
    if kind == "diversity":
        return {
            "component_set_hash": "cs123",
            "diversity_metric": "q_statistic",
            "split": "test",
        }
    if kind == "consistency":
        return {
            "component_set_hash": "cs123",
            "anchor_spec_hash": "a123",
            "split": "test",
        }
    if kind == "metalearner":
        return {
            "component_set_hash": "cs123",
            "split": "test",
            "feature_type": "embeddings+profiles",
            "anchor_spec": "a123",
            "meta_split_spec": '{"seed":42}',
            "similarity_metric": "cosine",
            "init_seed": "42",
            "profile_mask": "true_class",
            "meta_type": "meta_lr",
        }
    raise AssertionError(f"Unhandled kind: {kind}")


def test_registry_covers_all_schema_kinds():
    assert set(TELEMETRY_SCHEMA.keys()).issubset(set(IDEMPOTENCY_REGISTRY.keys()))
    assert set(IDEMPOTENCY_REGISTRY.keys()).issubset(set(TELEMETRY_SCHEMA.keys()))


@pytest.mark.parametrize("kind", sorted(IDEMPOTENCY_REGISTRY.keys()))
def test_identity_hash_changes_when_single_field_changes(kind: str):
    fields = _base_fields(kind)
    h1 = identity_hash(kind, **fields)
    one_key = next(iter(fields))
    changed = dict(fields)
    changed[one_key] = f"{changed[one_key]}_changed"
    h2 = identity_hash(kind, **changed)
    assert h1 != h2


def test_ensemble_method_collision_prevented():
    base = _base_fields("ensemble")
    soft = identity_hash("ensemble", **base)
    hard = identity_hash("ensemble", **{**base, "method": "hard"})
    assert soft != hard


def test_metalearner_meta_type_collision_prevented():
    base = _base_fields("metalearner")
    lr = identity_hash("metalearner", **base)
    mlp = identity_hash("metalearner", **{**base, "meta_type": "meta_mlp_2"})
    assert lr != mlp


def test_identity_hash_rejects_unknown_fields():
    with pytest.raises(ValueError, match="Unknown identity field"):
        identity_hash("ensemble", **{**_base_fields("ensemble"), "oops": "x"})
