"""
Tests that diagnostics identity hashes are split-sensitive.

Ensures that two diagnostic runs on the same model but different splits
(e.g. test vs val) produce distinct identity hashes, preventing accidental
deduplication across splits.
"""

from __future__ import annotations

from src.config.hash import identity_hash


_BASE = {
    "parent_run_id": "run_abc123",
    "diagnostic_metric": "morans_i",
    "split": "test",
}


def test_same_inputs_same_hash():
    h1 = identity_hash("diagnostics", **_BASE)
    h2 = identity_hash("diagnostics", **_BASE)
    assert h1 == h2


def test_different_splits_different_hash():
    h_test = identity_hash("diagnostics", **_BASE)
    h_val = identity_hash("diagnostics", **{**_BASE, "split": "val"})
    assert h_test != h_val


def test_different_metrics_different_hash():
    h1 = identity_hash("diagnostics", **_BASE)
    h2 = identity_hash("diagnostics", **{**_BASE, "diagnostic_metric": "weight_norms"})
    assert h1 != h2


def test_different_parent_run_different_hash():
    h1 = identity_hash("diagnostics", **_BASE)
    h2 = identity_hash("diagnostics", **{**_BASE, "parent_run_id": "run_xyz999"})
    assert h1 != h2
