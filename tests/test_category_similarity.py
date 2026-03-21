"""
Tests for the Category Similarity Profile pipeline.

Validates:
  - anchor_spec_hash stability and sensitivity
  - similarity_profile_hash stability and sensitivity
  - compute_similarity_profile correctness (cosine + L2)
  - demand-driven caching logic (mock MLflow)
  - behaviour_input_hash includes anchor_spec_hash + similarity_metric + feature_type
  - feature_type changes produced feature dimensions deterministically
  - category_similarity_profile_tags contain required keys
"""

from __future__ import annotations

import hashlib
import json
import os
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.data.anchors import AnchorSpec, anchor_spec_hash, select_anchors
from src.mlflow_utils import (
    behaviour_input_hash,
    category_similarity_profile_tags,
    component_set_hash,
    find_finished_similarity_profile_run,
)
from src.profiling.category_similarity import (
    compute_similarity_profile,
    similarity_profile_hash,
)


# ──────────── Fixtures ──────────────


def _make_labels(N=500, num_classes=10):
    return torch.tensor([i % num_classes for i in range(N)])


def _make_embeddings(N=500, D=256, seed=42):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(N, D, generator=gen)


# ──────── anchor_spec_hash ────────


class TestAnchorSpecHashStability:
    """anchor_spec_hash must be stable and sensitive to spec changes."""

    def test_stable_across_calls(self):
        spec = {
            "per_class": 100,
            "strategy": "per_class_first_n",
            "order_by": "example_id",
        }
        assert anchor_spec_hash(spec) == anchor_spec_hash(spec)

    def test_order_invariant(self):
        h1 = anchor_spec_hash({"a": 1, "b": 2, "c": 3})
        h2 = anchor_spec_hash({"c": 3, "a": 1, "b": 2})
        assert h1 == h2

    def test_per_class_changes_hash(self):
        base = {"per_class": 100, "strategy": "per_class_first_n"}
        changed = {"per_class": 200, "strategy": "per_class_first_n"}
        assert anchor_spec_hash(base) != anchor_spec_hash(changed)

    def test_strategy_changes_hash(self):
        h1 = anchor_spec_hash({"per_class": 100, "strategy": "per_class_first_n"})
        h2 = anchor_spec_hash({"per_class": 100, "strategy": "per_class_random"})
        assert h1 != h2

    def test_hash_length_16(self):
        h = anchor_spec_hash({"per_class": 50})
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)


# ──────── similarity_profile_hash ────────


class TestSimilarityProfileHash:
    """similarity_profile_hash must be stable and sensitive to all inputs."""

    def test_stable(self):
        h1 = similarity_profile_hash("run_abc", "anchor_hash", "cosine", "test")
        h2 = similarity_profile_hash("run_abc", "anchor_hash", "cosine", "test")
        assert h1 == h2

    def test_parent_run_changes_hash(self):
        h1 = similarity_profile_hash("run_AAA", "anchor_hash", "cosine", "test")
        h2 = similarity_profile_hash("run_BBB", "anchor_hash", "cosine", "test")
        assert h1 != h2

    def test_anchor_spec_changes_hash(self):
        h1 = similarity_profile_hash("run_abc", "anchor_AAA", "cosine", "test")
        h2 = similarity_profile_hash("run_abc", "anchor_BBB", "cosine", "test")
        assert h1 != h2

    def test_metric_changes_hash(self):
        h1 = similarity_profile_hash("run_abc", "anchor_hash", "cosine", "test")
        h2 = similarity_profile_hash("run_abc", "anchor_hash", "l2", "test")
        assert h1 != h2

    def test_split_changes_hash(self):
        h1 = similarity_profile_hash("run_abc", "anchor_hash", "cosine", "test")
        h2 = similarity_profile_hash("run_abc", "anchor_hash", "cosine", "val")
        assert h1 != h2

    def test_hash_length_16(self):
        h = similarity_profile_hash("r", "a", "cosine", "test")
        assert len(h) == 16


# ──────── compute_similarity_profile ────────


class TestComputeSimilarityProfile:
    """Test correctness of cosine and L2 similarity profile computation."""

    def test_cosine_shape(self):
        N, D, K = 100, 64, 20
        emb = torch.randn(N, D)
        anchors = torch.randn(K, D)
        profiles = compute_similarity_profile(emb, anchors, metric="cosine")
        assert profiles.shape == (N, K)

    def test_l2_shape(self):
        N, D, K = 100, 64, 20
        emb = torch.randn(N, D)
        anchors = torch.randn(K, D)
        profiles = compute_similarity_profile(emb, anchors, metric="l2")
        assert profiles.shape == (N, K)

    def test_cosine_self_similarity_is_one(self):
        """Cosine similarity of a vector with itself should be ~1.0."""
        emb = torch.randn(10, 32)
        profiles = compute_similarity_profile(emb, emb, metric="cosine")
        diagonal = torch.diag(profiles)
        assert torch.allclose(diagonal, torch.ones(10), atol=1e-5)

    def test_l2_self_distance_is_zero(self):
        """L2 distance of a vector with itself should be ~0 (profile = -0 ≈ 0)."""
        emb = torch.randn(10, 32)
        profiles = compute_similarity_profile(emb, emb, metric="l2")
        diagonal = torch.diag(profiles)
        assert torch.allclose(diagonal, torch.zeros(10), atol=5e-3)

    def test_cosine_range(self):
        """Cosine similarity should be in [-1, 1]."""
        emb = torch.randn(50, 64)
        anchors = torch.randn(20, 64)
        profiles = compute_similarity_profile(emb, anchors, metric="cosine")
        assert profiles.min() >= -1.0 - 1e-6
        assert profiles.max() <= 1.0 + 1e-6

    def test_l2_nonpositive(self):
        """Negated L2 should be <= 0."""
        emb = torch.randn(50, 64)
        anchors = torch.randn(20, 64)
        profiles = compute_similarity_profile(emb, anchors, metric="l2")
        assert profiles.max() <= 1e-6

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown similarity metric"):
            compute_similarity_profile(
                torch.randn(5, 3), torch.randn(2, 3), metric="dot"
            )

    def test_deterministic(self):
        """Same inputs produce identical outputs."""
        emb = torch.randn(20, 16)
        anchors = torch.randn(5, 16)
        p1 = compute_similarity_profile(emb, anchors, metric="cosine")
        p2 = compute_similarity_profile(emb, anchors, metric="cosine")
        assert torch.equal(p1, p2)


# ──────── behaviour_input_hash includes all identity fields ────────


class TestBehaviourInputHashIdempotency:
    """Meta-learner idempotency must include anchor_spec_hash + similarity_metric + feature_type."""

    def _base_args(self):
        return {
            "component_set_hash_val": component_set_hash(["run_a", "run_b"]),
            "split": "test",
            "feature_type": "embeddings+profiles",
            "anchor_spec": "anchor_hash_AAAA",
            "meta_split_spec": '{"seed": 42}',
            "similarity_metric": "cosine",
        }

    def test_same_inputs_same_hash(self):
        args = self._base_args()
        assert behaviour_input_hash(**args) == behaviour_input_hash(**args)

    def test_anchor_spec_changes_hash(self):
        args1 = self._base_args()
        args2 = {**args1, "anchor_spec": "anchor_hash_BBBB"}
        assert behaviour_input_hash(**args1) != behaviour_input_hash(**args2)

    def test_similarity_metric_changes_hash(self):
        args1 = self._base_args()
        args2 = {**args1, "similarity_metric": "l2"}
        assert behaviour_input_hash(**args1) != behaviour_input_hash(**args2)

    def test_feature_type_changes_hash(self):
        args1 = self._base_args()
        args2 = {**args1, "feature_type": "embeddings"}
        assert behaviour_input_hash(**args1) != behaviour_input_hash(**args2)

    def test_logits_vs_embeddings_different(self):
        args_logits = {**self._base_args(), "feature_type": "logits"}
        args_emb = {**self._base_args(), "feature_type": "embeddings"}
        assert behaviour_input_hash(**args_logits) != behaviour_input_hash(**args_emb)

    def test_empty_similarity_metric_backward_compat(self):
        """Empty similarity_metric is default for logits-only (backward compat)."""
        args = self._base_args()
        args["similarity_metric"] = ""
        h1 = behaviour_input_hash(**args)
        args["similarity_metric"] = "cosine"
        h2 = behaviour_input_hash(**args)
        assert h1 != h2


# ──────── feature_type dimensions ────────


class TestFeatureTypeDimensions:
    """Feature type must deterministically change produced feature dimensions."""

    def test_logits_dim(self):
        """Stacked logits: M models × C classes."""
        M, N, C = 3, 100, 10
        logits = [torch.randn(N, C) for _ in range(M)]
        stacked = torch.cat(logits, dim=1)
        assert stacked.shape == (N, M * C)

    def test_embeddings_dim(self):
        """Stacked embeddings: M models × D embedding_dim."""
        M, N, D = 5, 100, 256
        embs = [torch.randn(N, D) for _ in range(M)]
        stacked = torch.cat(embs, dim=1)
        assert stacked.shape == (N, M * D)

    def test_embeddings_plus_profiles_dim(self):
        """Stacked embeddings+profiles: M models × (D + K)."""
        M, N, D, K = 5, 100, 256, 50
        features = []
        for _ in range(M):
            emb = torch.randn(N, D)
            prof = torch.randn(N, K)
            features.append(torch.cat([emb, prof], dim=1))
        stacked = torch.cat(features, dim=1)
        assert stacked.shape == (N, M * (D + K))

    def test_profiles_add_extra_dims(self):
        """embeddings+profiles must have more features than embeddings alone."""
        M, N, D, K = 3, 100, 256, 50
        emb_only = torch.cat([torch.randn(N, D) for _ in range(M)], dim=1)
        ep = torch.cat(
            [
                torch.cat([torch.randn(N, D), torch.randn(N, K)], dim=1)
                for _ in range(M)
            ],
            dim=1,
        )
        assert ep.shape[1] == emb_only.shape[1] + M * K


# ──────── category_similarity_profile_tags ────────


class TestCategorySimilarityProfileTags:
    """Tags for CSP runs must contain all required identity fields."""

    def test_required_keys(self):
        tags = category_similarity_profile_tags(
            parent_run_id="run_abc",
            anchor_spec_hash="hash_123",
            similarity_metric="cosine",
            split="test",
            profile_hash="prof_abc",
        )
        assert tags["kind"] == "category_similarity_profile"
        assert tags["parent_run_id"] == "run_abc"
        assert tags["anchor_spec_hash"] == "hash_123"
        assert tags["similarity_metric"] == "cosine"
        assert tags["split"] == "test"
        assert tags["profile_hash"] == "prof_abc"

    def test_extra_tags_merged(self):
        tags = category_similarity_profile_tags(
            parent_run_id="r",
            anchor_spec_hash="a",
            similarity_metric="l2",
            split="val",
            profile_hash="p",
            extra={"rho": "0.04", "trial": "2"},
        )
        assert tags["rho"] == "0.04"
        assert tags["trial"] == "2"
        assert tags["kind"] == "category_similarity_profile"


# ──────── demand-driven caching (mock MLflow) ────────


class TestDemandDrivenCaching:
    """Verify demand-driven caching logic: compute only on cache miss."""

    @patch("src.mlflow_utils.mlflow")
    def test_find_finished_returns_none_on_miss(self, mock_mlflow):
        """When no matching run exists, find_finished_similarity_profile_run returns None."""
        mock_mlflow.get_experiment_by_name.return_value = None
        result = find_finished_similarity_profile_run(
            "test_exp", "run_abc", "hash_123", "cosine", "test"
        )
        assert result is None

    @patch("src.mlflow_utils.mlflow")
    def test_find_finished_returns_run_on_hit(self, mock_mlflow):
        """When a matching run exists, it is returned."""
        mock_exp = MagicMock()
        mock_exp.experiment_id = "exp_1"
        mock_mlflow.get_experiment_by_name.return_value = mock_exp

        mock_run = MagicMock()
        mock_run.info.run_id = "cached_run_id"
        mock_mlflow.search_runs.return_value = [mock_run]

        result = find_finished_similarity_profile_run(
            "test_exp", "run_abc", "hash_123", "cosine", "test"
        )
        assert result is not None
        assert result.info.run_id == "cached_run_id"

        # Verify the filter string contains all identity fields
        call_kwargs = mock_mlflow.search_runs.call_args
        filter_str = call_kwargs.kwargs.get("filter_string") or call_kwargs[1].get(
            "filter_string", ""
        )
        assert "category_similarity_profile" in filter_str
        assert "run_abc" in filter_str
        assert "hash_123" in filter_str
        assert "cosine" in filter_str

    def test_local_cache_saves_and_loads(self, tmp_path):
        """Profiles saved locally can be loaded back identically."""
        profiles = torch.randn(100, 50)
        path = str(tmp_path / "profiles.pt")
        torch.save(profiles, path)
        loaded = torch.load(path, weights_only=True)
        assert torch.equal(profiles, loaded)


# ──────── end-to-end anchor → profile → feature assembly (unit) ────────


class TestEndToEndFeatureAssembly:
    """Integration test: anchor selection → profile computation → feature assembly."""

    def test_full_pipeline_deterministic(self):
        """Given fixed labels + embeddings, the entire pipeline is deterministic."""
        N, D, K_per_class, num_classes = 500, 64, 5, 10
        labels = _make_labels(N=N, num_classes=num_classes)
        emb = _make_embeddings(N=N, D=D, seed=42)

        # Select anchors
        spec = AnchorSpec(
            source_split="test",
            per_class=K_per_class,
            strategy="per_class_first_n",
            order_by="original_index",
            num_classes=num_classes,
        )
        anchors = select_anchors(labels, spec)
        K = K_per_class * num_classes  # total anchors
        assert len(anchors["anchor_indices"]) == K

        # Compute profiles
        anchor_emb = emb[anchors["anchor_indices"]]
        profiles_cos = compute_similarity_profile(emb, anchor_emb, metric="cosine")
        profiles_l2 = compute_similarity_profile(emb, anchor_emb, metric="l2")

        assert profiles_cos.shape == (N, K)
        assert profiles_l2.shape == (N, K)

        # Feature assembly: embeddings+profiles
        combined = torch.cat([emb, profiles_cos], dim=1)
        assert combined.shape == (N, D + K)

        # Determinism: repeat and compare
        anchors2 = select_anchors(labels, spec)
        anchor_emb2 = emb[anchors2["anchor_indices"]]
        profiles_cos2 = compute_similarity_profile(emb, anchor_emb2, metric="cosine")
        assert torch.equal(profiles_cos, profiles_cos2)
        assert anchors["spec_hash"] == anchors2["spec_hash"]
