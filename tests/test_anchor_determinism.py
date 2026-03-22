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
    compute_anchor_spec_hash,
    get_anchor_spec_dict,
    select_anchors,
)
from src.mlflow_utils import behaviour_input_hash, component_set_hash


def _make_test_labels(N=500, num_classes=10):
    """Create synthetic ground-truth labels for testing."""
    return torch.tensor([i % num_classes for i in range(N)])


def _make_spec(**overrides) -> dict:
    """Create a spec dictionary with sensible test defaults."""
    defaults = dict(
        source_split="test",
        per_class=5,
        strategy="per_class_first_n",
        order_by="original_index",
        num_classes=10,
    )
    defaults.update(overrides)
    return defaults


class TestAnchorSpecCreation:
    """Test the functional deterministic hashing of anchor specs."""

    def test_hash_stable(self):
        spec = _make_spec()
        assert compute_anchor_spec_hash(**spec) == compute_anchor_spec_hash(**spec)

    def test_hash_length(self):
        spec = _make_spec()
        assert len(compute_anchor_spec_hash(**spec)) == 16

    def test_different_per_class_different_hash(self):
        s1 = _make_spec(per_class=5)
        s2 = _make_spec(per_class=10)
        assert compute_anchor_spec_hash(**s1) != compute_anchor_spec_hash(**s2)

    def test_to_dict_helper(self):
        spec = _make_spec()
        d = get_anchor_spec_dict(**spec)
        assert isinstance(d, dict)
        assert d["per_class"] == 5
        assert d["source_split"] == "test"


class TestAnchorSelection:
    """Test deterministic anchor selection from labels."""

    def test_deterministic(self):
        labels = _make_test_labels()
        spec = _make_spec(per_class=5)
        a1 = select_anchors(labels, **spec)
        a2 = select_anchors(labels, **spec)
        assert a1["anchor_indices"] == a2["anchor_indices"]

    def test_correct_count(self):
        labels = _make_test_labels()
        anchors = select_anchors(labels, **_make_spec(per_class=5))
        assert len(anchors["anchor_indices"]) == 50  # 5 * 10

    def test_spec_hash_in_result(self):
        labels = _make_test_labels()
        anchors = select_anchors(labels, **_make_spec(per_class=5))
        assert "spec_hash" in anchors
        assert len(anchors["spec_hash"]) == 16

    def test_different_per_class_different_result(self):
        labels = _make_test_labels()
        a1 = select_anchors(labels, **_make_spec(per_class=5))
        a2 = select_anchors(labels, **_make_spec(per_class=3))
        assert a1["spec_hash"] != a2["spec_hash"]
        assert len(a2["anchor_indices"]) == 30

    def test_raises_on_insufficient_samples(self):
        labels = _make_test_labels(N=50, num_classes=10)
        with pytest.raises(RuntimeError, match="only .* examples"):
            select_anchors(labels, **_make_spec(per_class=100))


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
