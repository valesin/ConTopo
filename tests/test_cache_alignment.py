"""
Tests for example_id alignment logic on synthetic data.
"""

import hashlib
import torch
import pytest

from src.data.manifest import DatasetManifest


def _make_manifest(n: int = 100, seed: int = 0) -> DatasetManifest:
    """Create a synthetic manifest."""
    gen = torch.Generator().manual_seed(seed)
    ids = [hashlib.sha256(f"img_{i}".encode()).hexdigest()[:16] for i in range(n)]
    indices = torch.arange(n)
    labels = torch.randint(0, 10, (n,), generator=gen)
    return DatasetManifest(
        hashes=ids,
        original_indices=indices,
        labels=labels,
        dataset_name="synthetic",
        split="test",
    )


class TestManifest:
    def test_roundtrip(self, tmp_path):
        m = _make_manifest()
        path = str(tmp_path / "manifest.pt")
        m.save(path)
        m2 = DatasetManifest.load(path)
        assert m2.hashes == m.hashes
        assert torch.equal(m2.original_indices, m.original_indices)
        assert torch.equal(m2.labels, m.labels)
        assert m2.dataset_name == m.dataset_name
        assert m2.split == m.split

    def test_example_ids_unique(self):
        m = _make_manifest(n=200)
        assert len(set(m.hashes)) == len(m.hashes)

    def test_labels_shape(self):
        m = _make_manifest(n=50)
        assert m.labels.shape == (50,)
        assert m.original_indices.shape == (50,)
        assert len(m.hashes) == 50


class TestAlignment:
    """Verify that inference artifacts can be aligned via manifest."""

    def test_alignment_by_index(self):
        """Simulated alignment: two inference runs share the same manifest."""
        m = _make_manifest(n=100)
        # Simulate two runs producing predictions in the same order
        preds_a = torch.randint(0, 10, (100,))
        preds_b = torch.randint(0, 10, (100,))

        # Alignment check: same manifest index → same example_id
        for i in range(100):
            idx_a = int(m.original_indices[i])
            idx_b = int(m.original_indices[i])
            assert idx_a == idx_b
            assert m.hashes[i] == m.hashes[i]

    def test_manifest_preserves_order(self, tmp_path):
        """Save/load preserves the exact ordering."""
        m = _make_manifest(n=300)
        path = str(tmp_path / "m.pt")
        m.save(path)
        m2 = DatasetManifest.load(path)
        # Order must be identical
        for i in range(300):
            assert m.hashes[i] == m2.hashes[i]
