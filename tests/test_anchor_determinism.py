"""
Tests for anchor selection determinism and specification hashing.

Validates:
  - Anchor selection is deterministic given the same manifest + spec
  - anchor_spec_hash is deterministic and sensitive to spec changes
  - Anchor identity (spec_hash) can be included in behavior_input_hash
"""

from __future__ import annotations

import pytest
import torch

from src.data.anchors import (
    anchor_spec_hash,
    select_anchors_from_manifest,
)
from src.data.manifest import DatasetManifest
from src.mlflow_utils import behavior_input_hash, component_set_hash


def _make_test_manifest(N=500, num_classes=10, split="test"):
    """Create a synthetic DatasetManifest for testing."""
    labels = torch.tensor([i % num_classes for i in range(N)])
    example_ids = [f"img_{i:05d}" for i in range(N)]
    original_indices = torch.arange(N)
    return DatasetManifest(
        example_ids=example_ids,
        original_indices=original_indices,
        labels=labels,
        dataset_name="test",
        split=split,
    )


class TestAnchorSpecHash:
    """Test deterministic hashing of anchor specifications."""

    def test_stable(self):
        spec = {"per_class": 100, "strategy": "per_class_first_n", "order_by": "example_id"}
        h1 = anchor_spec_hash(spec)
        h2 = anchor_spec_hash(spec)
        assert h1 == h2

    def test_order_invariant(self):
        h1 = anchor_spec_hash({"a": 1, "b": 2})
        h2 = anchor_spec_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_different_spec_different_hash(self):
        h1 = anchor_spec_hash({"per_class": 100})
        h2 = anchor_spec_hash({"per_class": 200})
        assert h1 != h2

    def test_hash_length(self):
        h = anchor_spec_hash({"per_class": 50})
        assert len(h) == 16


class TestAnchorSelection:
    """Test deterministic anchor selection from manifest."""

    def test_deterministic(self):
        manifest = _make_test_manifest()
        a1 = select_anchors_from_manifest(manifest, per_class=5, num_classes=10)
        a2 = select_anchors_from_manifest(manifest, per_class=5, num_classes=10)
        assert a1["anchor_indices"] == a2["anchor_indices"]
        assert a1["anchor_example_ids"] == a2["anchor_example_ids"]

    def test_correct_count(self):
        manifest = _make_test_manifest()
        anchors = select_anchors_from_manifest(manifest, per_class=5, num_classes=10)
        assert len(anchors["anchor_indices"]) == 50  # 5 * 10

    def test_spec_hash_in_result(self):
        manifest = _make_test_manifest()
        anchors = select_anchors_from_manifest(manifest, per_class=5, num_classes=10)
        assert "spec_hash" in anchors
        assert len(anchors["spec_hash"]) == 16

    def test_different_per_class_different_result(self):
        manifest = _make_test_manifest()
        a1 = select_anchors_from_manifest(manifest, per_class=5, num_classes=10)
        a2 = select_anchors_from_manifest(manifest, per_class=3, num_classes=10)
        assert a1["spec_hash"] != a2["spec_hash"]
        assert len(a2["anchor_indices"]) == 30

    def test_raises_on_insufficient_samples(self):
        manifest = _make_test_manifest(N=50, num_classes=10)
        with pytest.raises(RuntimeError, match="only .* examples"):
            select_anchors_from_manifest(manifest, per_class=100, num_classes=10)


class TestAnchorIdentityInBehaviorHash:
    """Test that anchor spec hash affects behavior_input_hash (requirement #4)."""

    def test_anchor_spec_changes_behavior_hash(self):
        """Changing anchor selection MUST change meta-learner behavior_input_hash."""
        cs = component_set_hash(["run_a", "run_b"])
        meta_split = '{"seed": 42}'

        h1 = behavior_input_hash(
            cs, split="test", feature_type="logits",
            anchor_spec="anchor_hash_AAA",
            meta_split_spec=meta_split,
        )
        h2 = behavior_input_hash(
            cs, split="test", feature_type="logits",
            anchor_spec="anchor_hash_BBB",
            meta_split_spec=meta_split,
        )
        assert h1 != h2

    def test_same_anchors_same_behavior_hash(self):
        """Same anchor spec MUST produce same behavior_input_hash."""
        cs = component_set_hash(["run_a", "run_b"])
        meta_split = '{"seed": 42}'

        h1 = behavior_input_hash(
            cs, split="test", feature_type="logits",
            anchor_spec="anchor_hash_SAME",
            meta_split_spec=meta_split,
        )
        h2 = behavior_input_hash(
            cs, split="test", feature_type="logits",
            anchor_spec="anchor_hash_SAME",
            meta_split_spec=meta_split,
        )
        assert h1 == h2
