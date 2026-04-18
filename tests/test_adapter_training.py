from __future__ import annotations

import copy
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.training.adapter_training import (
    three_way_split,
    standardize_features,
    train_adapter,
    evaluate_holdout,
)
from src.networks.heads import LinearAdapter


def test_three_way_split_determinism():
    fractions = {"train": 0.5, "val": 0.25, "holdout": 0.25}
    t1, v1, h1 = three_way_split(100, fractions, seed=42)
    t2, v2, h2 = three_way_split(100, fractions, seed=42)

    assert np.array_equal(t1, t2)
    assert np.array_equal(v1, v2)
    assert np.array_equal(h1, h2)


def test_three_way_split_coverage():
    fractions = {"train": 0.4, "val": 0.3, "holdout": 0.3}
    t, v, h = three_way_split(100, fractions, seed=123)

    all_indices = np.concatenate([t, v, h])
    assert len(all_indices) == 100
    assert len(np.unique(all_indices)) == 100


def test_standardize_features_zero_mean_unit_std():
    # Simulate data
    torch.manual_seed(0)
    train_feat = torch.randn(100, 10) * 5 + 10
    val_feat = torch.randn(50, 10) * 5 + 10
    holdout_feat = torch.randn(50, 10) * 5 + 10

    t_s, v_s, h_s = standardize_features(train_feat, val_feat, holdout_feat)

    # Train means should be ~0 and stds ~1
    assert torch.allclose(t_s.mean(dim=0), torch.zeros(10), atol=1e-5)
    assert torch.allclose(t_s.std(dim=0), torch.ones(10), atol=1e-4)

    # Validation should be roughly scaled by train norm, but not exactly 0.
    assert not torch.allclose(v_s.mean(dim=0), torch.zeros(10), atol=1e-5)


def test_train_adapter_improves_over_random():
    # Setup data: signal is in the first two dimensions
    torch.manual_seed(42)
    X_train = torch.randn(200, 5)
    y_train = ((X_train[:, 0] > 0).long() + (X_train[:, 1] > 0).long()).clamp(0, 2)

    X_val = torch.randn(50, 5)
    y_val = ((X_val[:, 0] > 0).long() + (X_val[:, 1] > 0).long()).clamp(0, 2)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=32)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = LinearAdapter(emb_dim=5, num_classes=3)
    initial_model = copy.deepcopy(model)

    criterion = nn.CrossEntropyLoss()

    _, initial_acc, _ = evaluate_holdout(
        initial_model, torch.device("cpu"), val_loader, criterion
    )

    trained_model, history = train_adapter(
        model,
        device=torch.device("cpu"),
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=15,
        lr=0.05,
    )

    _, final_acc, _ = evaluate_holdout(
        trained_model, torch.device("cpu"), val_loader, criterion
    )
    assert len(history) == 15
    assert final_acc > initial_acc
