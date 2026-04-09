from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "05_train_adapters.py"
_SPEC = importlib.util.spec_from_file_location("train_adapters_script", _SCRIPT_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
_apply_profile_mask = _MODULE._apply_profile_mask


def _expected_masked(P: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
    N, M, C = P.shape
    out = []
    for n in range(N):
        cls = int(preds[n].item())
        keep = [c for c in range(C) if c != cls]
        out.append(P[n, :, keep])
    return torch.stack(out, dim=0)


def test_argmax_similarity_uses_component_mean_and_masks_same_class_across_components():
    P = torch.tensor(
        [
            [
                [0.1, 0.7, 0.2, 0.0],
                [0.1, 0.1, 0.8, 0.0],
                [0.1, 0.7, 0.2, 0.0],
            ],
            [
                [0.8, 0.1, 0.1, 0.0],
                [0.2, 0.1, 0.7, 0.0],
                [0.2, 0.1, 0.7, 0.0],
            ],
        ],
        dtype=torch.float32,
    )

    labels = torch.tensor([0, 1])
    component_logits = [torch.zeros(2, 4) for _ in range(3)]

    got = _apply_profile_mask(
        P=P,
        mask_type="argmax_similarity",
        labels=labels,
        component_logit_preds=component_logits,
        indices=None,
    )

    preds = P.mean(dim=1).argmax(dim=1)
    expected = _expected_masked(P, preds)

    assert got.shape == (2, 3, 3)
    assert torch.equal(got, expected)


def test_argmax_logits_uses_raw_component_mean_with_subset_indices():
    P_subset = torch.tensor(
        [
            [
                [10.0, 11.0, 12.0],
                [20.0, 21.0, 22.0],
                [30.0, 31.0, 32.0],
            ],
            [
                [40.0, 41.0, 42.0],
                [50.0, 51.0, 52.0],
                [60.0, 61.0, 62.0],
            ],
        ],
        dtype=torch.float32,
    )

    logits_comp_1 = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            [3.0, 1.0, 3.0],
        ],
        dtype=torch.float32,
    )
    logits_comp_2 = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            [3.0, 2.0, 2.0],
        ],
        dtype=torch.float32,
    )
    logits_comp_3 = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            [3.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    indices = np.array([1, 3], dtype=np.int64)
    labels = torch.tensor([0, 1])

    got = _apply_profile_mask(
        P=P_subset,
        mask_type="argmax_logits",
        labels=labels,
        component_logit_preds=[logits_comp_1, logits_comp_2, logits_comp_3],
        indices=indices,
    )

    stacked = torch.stack(
        [logits_comp_1[indices], logits_comp_2[indices], logits_comp_3[indices]], dim=1
    )
    preds = stacked.mean(dim=1).argmax(dim=1)
    expected = _expected_masked(P_subset, preds)

    assert got.shape == (2, 3, 2)
    assert torch.equal(got, expected)


def test_argmax_logits_raises_on_non_finite_logits():
    P = torch.ones(1, 2, 3)
    labels = torch.tensor([0])
    logits_ok = torch.tensor([[1.0, 2.0, 3.0]])
    logits_bad = torch.tensor([[1.0, float("nan"), 3.0]])

    with pytest.raises(ValueError, match="NaN/Inf"):
        _apply_profile_mask(
            P=P,
            mask_type="argmax_logits",
            labels=labels,
            component_logit_preds=[logits_ok, logits_bad],
            indices=None,
        )


def test_profile_mask_raises_on_non_finite_profiles():
    P = torch.tensor([[[1.0, float("inf"), 3.0]]])
    labels = torch.tensor([0])

    with pytest.raises(ValueError, match="NaN/Inf"):
        _apply_profile_mask(
            P=P,
            mask_type="argmax_similarity",
            labels=labels,
            component_logit_preds=[torch.zeros(1, 3)],
            indices=None,
        )
