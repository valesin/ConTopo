from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def three_way_split(N: int, fractions: dict, seed: int):
    """Deterministically split N indices into train/val/holdout."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(N)

    f_train = fractions.get("train", 0.6)
    f_val = fractions.get("val", 0.2)

    n_train = int(N * f_train)
    n_val = int(N * f_val)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    holdout_idx = indices[n_train + n_val :]

    return train_idx, val_idx, holdout_idx

def standardize_features(
    train_feat: torch.Tensor, val_feat: torch.Tensor, holdout_feat: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standardize features using mean and std computed ONLY on the training set
    to prevent data leakage into the validation and holdout sets.
    """
    mean = train_feat.mean(dim=0, keepdim=True)
    std = train_feat.std(dim=0, keepdim=True)

    # Add epsilon to prevent division by zero
    train_std = (train_feat - mean) / (std + 1e-6)
    val_std = (val_feat - mean) / (std + 1e-6)
    holdout_std = (holdout_feat - mean) / (std + 1e-6)

    return train_std, val_std, holdout_std

def train_adapter(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
) -> tuple[nn.Module, list[dict]]:
    """Train adapter loop keeping track of the best validation accuracy parameters."""
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_val_acc = -1.0
    best_state: dict | None = None

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimiser.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimiser.step()

            train_loss += loss.item() * X.size(0)
            train_correct += (torch.argmax(out, dim=1) == y).sum().item()
            train_total += X.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = criterion(out, y)
                val_loss += loss.item() * X.size(0)
                val_correct += (torch.argmax(out, dim=1) == y).sum().item()
                val_total += X.size(0)

        val_acc = val_correct / max(val_total, 1)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(train_total, 1),
                "train_acc": train_correct / max(train_total, 1),
                "val_loss": val_loss / max(val_total, 1),
                "val_acc": val_acc,
            }
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history

def evaluate_holdout(
    model: nn.Module, device: torch.device, loader: DataLoader, criterion: nn.Module
) -> tuple[float, float, torch.Tensor]:
    """Evaluate trained model on holdout set returning loss, accuracy, and output probabilities."""
    model.eval()
    holdout_loss, holdout_corr, holdout_tot = 0.0, 0, 0
    all_probs = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            probs = torch.softmax(out, dim=1)
            all_probs.append(probs.cpu())
            loss = criterion(out, y)
            holdout_loss += loss.item() * X.size(0)
            holdout_corr += (torch.argmax(out, dim=1) == y).sum().item()
            holdout_tot += X.size(0)

    mean_loss = holdout_loss / max(holdout_tot, 1)
    acc = holdout_corr / max(holdout_tot, 1)
    return mean_loss, acc, torch.cat(all_probs, dim=0)
