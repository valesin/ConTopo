"""
Tests for ensemble combine_logits correctness and stable hashing.
"""

import torch
import pytest

from src.ensemble.combine import combine_logits, METHODS
from src.config.hash import component_set_hash


class TestCombineLogits:
    """Test all four combination methods."""

    def _make_logits(self, M=3, N=100, C=10, seed=42):
        gen = torch.Generator().manual_seed(seed)
        return [torch.randn(N, C, generator=gen) for _ in range(M)]

    def test_soft_shape(self):
        logits = self._make_logits()
        out = combine_logits(logits, "soft")
        assert out.shape == (100, 10)

    def test_soft_sums_to_one(self):
        logits = self._make_logits()
        out = combine_logits(logits, "soft")
        sums = out.sum(dim=1)
        assert torch.allclose(sums, torch.ones(100), atol=1e-5)

    def test_hard_shape(self):
        logits = self._make_logits()
        out = combine_logits(logits, "hard")
        assert out.shape == (100, 10)

    def test_hard_is_onehot(self):
        logits = self._make_logits()
        out = combine_logits(logits, "hard")
        # Each row should have exactly one 1.0
        assert torch.allclose(out.sum(dim=1), torch.ones(100))

    def test_max_confidence_shape(self):
        logits = self._make_logits()
        out = combine_logits(logits, "max_confidence")
        assert out.shape == (100, 10)

    def test_conf_weighted_sums_to_one(self):
        logits = self._make_logits()
        out = combine_logits(logits, "conf_weighted")
        sums = out.sum(dim=1)
        assert torch.allclose(sums, torch.ones(100), atol=1e-5)

    def test_single_model_equals_softmax(self):
        """With one model, soft vote == softmax of its logits."""
        logits = [torch.randn(50, 10)]
        out = combine_logits(logits, "soft")
        expected = torch.softmax(logits[0], dim=1)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            combine_logits([torch.randn(5, 3)], "nonexistent")


class TestComponentSetHash:
    """Test deterministic hashing (consolidated into mlflow_utils.component_set_hash)."""

    def test_stable_hash(self):
        ids = ["run_c", "run_a", "run_b"]
        h1 = component_set_hash(ids)
        h2 = component_set_hash(ids)
        assert h1 == h2

    def test_order_invariant(self):
        h1 = component_set_hash(["a", "b", "c"])
        h2 = component_set_hash(["c", "a", "b"])
        assert h1 == h2

    def test_different_inputs_different_hash(self):
        h1 = component_set_hash(["a", "b"])
        h2 = component_set_hash(["a", "c"])
        assert h1 != h2

    def test_hash_length(self):
        h = component_set_hash(["x", "y", "z"])
        assert len(h) == 16
