"""
Tests for anchor selection determinism and specification hashing.

Validates:
  - AnchorSpec dataclass and its hash property
  - Anchor selection is deterministic given the same labels + spec
  - anchor_spec_hash is deterministic and sensitive to spec changes
  - Anchor identity (spec_hash) can be included in behaviour_input_hash
"""

from __future__ import annotations

import pytest
import torch

from src.data.anchors import (
    AnchorSpec,
    anchor_spec_hash,
    select_anchors,
)
from src.mlflow_utils import behaviour_input_hash, component_set_hash


def _make_test_labels(N=500, num_classes=10):
    """Create synthetic ground-truth labels for testing."""
    return torch.tensor([i % num_classes for i in range(N)])


def _make_spec(**overrides) -> AnchorSpec:
    """Create an AnchorSpec with sensible test defaults."""
    defaults = dict(
        source_split="test",
        per_class=5,
        strategy="per_class_first_n",
        order_by="original_index",
        num_classes=10,
    )
    defaults.update(overrides)
    return AnchorSpec(**defaults)


class TestAnchorSpec:
    """Test the AnchorSpec dataclass itself."""

    def test_frozen(self):
        spec = _make_spec()
        with pytest.raises(AttributeError):
            spec.per_class = 200  # type: ignore[misc]

    def test_hash_stable(self):
        spec = _make_spec()
        assert spec.hash == spec.hash

    def test_hash_length(self):
        assert len(_make_spec().hash) == 16

    def test_different_per_class_different_hash(self):
        h1 = _make_spec(per_class=5).hash
        h2 = _make_spec(per_class=10).hash
        assert h1 != h2

    def test_to_dict(self):
        spec = _make_spec()
        d = spec.to_dict()
        assert isinstance(d, dict)
        assert d["per_class"] == 5
        assert d["source_split"] == "test"

    def test_hash_matches_legacy_dict_hash(self):
        """AnchorSpec.hash must agree with anchor_spec_hash(dict)."""
        spec = _make_spec()
        assert spec.hash == anchor_spec_hash(spec.to_dict())


class TestAnchorSpecHash:
    """Test deterministic hashing of anchor specifications (legacy dict API)."""

    def test_stable(self):
        spec = {
            "per_class": 100,
            "strategy": "per_class_first_n",
            "order_by": "original_index",
        }
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
    """Test deterministic anchor selection from labels."""

    def test_deterministic(self):
        labels = _make_test_labels()
        spec = _make_spec(per_class=5)
        a1 = select_anchors(labels, spec)
        a2 = select_anchors(labels, spec)
        assert a1["anchor_indices"] == a2["anchor_indices"]

    def test_correct_count(self):
        labels = _make_test_labels()
        anchors = select_anchors(labels, _make_spec(per_class=5))
        assert len(anchors["anchor_indices"]) == 50  # 5 * 10

    def test_spec_hash_in_result(self):
        labels = _make_test_labels()
        anchors = select_anchors(labels, _make_spec(per_class=5))
        assert "spec_hash" in anchors
        assert len(anchors["spec_hash"]) == 16

    def test_different_per_class_different_result(self):
        labels = _make_test_labels()
        a1 = select_anchors(labels, _make_spec(per_class=5))
        a2 = select_anchors(labels, _make_spec(per_class=3))
        assert a1["spec_hash"] != a2["spec_hash"]
        assert len(a2["anchor_indices"]) == 30

    def test_raises_on_insufficient_samples(self):
        labels = _make_test_labels(N=50, num_classes=10)
        with pytest.raises(RuntimeError, match="only .* examples"):
            select_anchors(labels, _make_spec(per_class=100))


class TestAnchorIdentityInBehaviourHash:
    """Test that anchor spec hash affects behaviour_input_hash (requirement #4)."""

    def test_anchor_spec_changes_behaviour_hash(self):
        """Changing anchor selection MUST change meta-learner behaviour_input_hash."""
        cs = component_set_hash(["run_a", "run_b"])
        meta_split = '{"seed": 42}'

        h1 = behaviour_input_hash(
            cs,
            split="test",
            feature_type="logits",
            anchor_spec="anchor_hash_AAA",
            meta_split_spec=meta_split,
        )
        h2 = behaviour_input_hash(
            cs,
            split="test",
            feature_type="logits",
            anchor_spec="anchor_hash_BBB",
            meta_split_spec=meta_split,
        )
        assert h1 != h2

    def test_same_anchors_same_behaviour_hash(self):
        """Same anchor spec MUST produce same behaviour_input_hash."""
        cs = component_set_hash(["run_a", "run_b"])
        meta_split = '{"seed": 42}'

        h1 = behaviour_input_hash(
            cs,
            split="test",
            feature_type="logits",
            anchor_spec="anchor_hash_SAME",
            meta_split_spec=meta_split,
        )
        h2 = behaviour_input_hash(
            cs,
            split="test",
            feature_type="logits",
            anchor_spec="anchor_hash_SAME",
            meta_split_spec=meta_split,
        )
        assert h1 == h2
