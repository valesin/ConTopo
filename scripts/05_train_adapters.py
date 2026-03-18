#!/usr/bin/env python3
"""
05_train_adapters.py — Meta-learner / adapter training over ensemble components.

This script operates on FINISHED component runs dynamically grouped.
It learns a meta-combination (e.g., Logistic Regression or MLP) over the
frozen representations (logits or embeddings). Topographic profiles can be
appended as additional features.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

import hydra
import mlflow
from mlflow.models.signature import infer_signature
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from src.data.manifest import get_or_create_manifest
from src.ensemble.selector import discover_ensembles
from src.config.paths import get_cache_dir
from src.mlflow_utils import (
    behaviour_tags,
    component_set_hash,
    find_finished_metalearner_run,
    behaviour_input_hash,
    log_resolved_config,
    setup_mlflow,
    get_inference_run,
    get_profile_run,
    load_mlflow_artifact,
    log_manifest_lineage,
    safe_to_numpy_float64,
)

# ─── MODELS ───


class MetaLR(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes, bias=bias)

    def forward(self, x):
        return self.linear(x)


class AdapterMLP(nn.Module):

    def __init__(self, input_dim: int, num_classes: int, bias: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128, bias=bias),
            nn.ReLU(),
            nn.Linear(128, 64, bias=bias),
            nn.ReLU(),
            nn.Linear(64, num_classes, bias=bias),
        )

    def forward(self, x):
        return self.net(x)


# ─── LOGIC ───


def _three_way_split(N: int, fractions: dict, seed: int):
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


def _standardize_features(
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


def _train_adapter(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
) -> tuple[nn.Module, list[dict]]:
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = []

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

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(train_total, 1),
                "train_acc": train_correct / max(train_total, 1),
                "val_loss": val_loss / max(val_total, 1),
                "val_acc": val_correct / max(val_total, 1),
            }
        )
    return model, history


def _evaluate_holdout(
    model, device, loader, criterion
) -> tuple[float, float, torch.Tensor]:
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


def _assert_valid_feature_tensor(
    name: str, tensor: torch.Tensor, expected_rows: int
) -> None:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={tuple(tensor.shape)}")
    if tensor.shape[0] != expected_rows:
        raise ValueError(
            f"{name} row mismatch: expected {expected_rows}, got {tensor.shape[0]}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN/Inf values")


# ─── MAIN ───


def main():
    _main()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def _main(cfg: DictConfig) -> None:
    setup_mlflow(cfg)

    split = cfg.pipeline.split
    cache_dir = get_cache_dir(cfg)

    init_seed = cfg.adapter.init_seed
    torch.manual_seed(init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(init_seed)

    adapter_epochs = cfg.adapter.epochs
    adapter_lr = cfg.adapter.learning_rate
    adapter_batch_size = cfg.adapter.batch_size
    adapter_bias = cfg.adapter.bias

    meta_type = cfg.adapter.meta_type
    feature_type = cfg.adapter.feature_type
    similarity_metric = cfg.adapter.similarity_metric
    meta_split_cfg = cfg.adapter.meta_split
    use_profiles = "profiles" in feature_type

    # 1. Dataset Manifest Lock
    manifest = get_or_create_manifest(
        dataset_name=cfg.dataset.name,
        split=split,
        data_root=cfg.runtime.data_root,
        artifacts_root=str(cache_dir),
    )
    labels_tensor = manifest.labels
    total_examples = len(labels_tensor)

    # 2. Discover Ensemble Components from MLflow
    groups = discover_ensembles(cfg.mlflow.experiment_name)
    if not groups:
        print("No dynamic ensembles discovered. Exiting.")
        return

    client = mlflow.tracking.MlflowClient()
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        and getattr(cfg.pipeline, "device", "cuda") != "cpu"
        else "cpu"
    )

    for ens_name, run_ids in groups.items():
        print(
            f"\n{'=' * 60}\nAdapters: {ens_name} | Meta: {meta_type} | Feat: {feature_type}"
        )

        cs_hash = component_set_hash(run_ids)
        adapter_cfg_dict = OmegaConf.to_container(cfg.adapter, resolve=True)
        # Lock identity
        bi_hash = behaviour_input_hash(
            component_set_hash_val=cs_hash,
            dataset_manifest_hash=manifest.manifest_hash,
            split=split,
            feature_type=feature_type,
            similarity_metric=similarity_metric if use_profiles else "",
            init_seed=str(init_seed)
            + str(meta_split_cfg.seed)
            + str(meta_split_cfg.fractions),
        )

        existing = find_finished_metalearner_run(
            cfg.mlflow.experiment_name, bi_hash, meta_type=meta_type
        )
        if existing is not None and not cfg.pipeline.force:
            print(
                f"  SKIP: {meta_type} adapter already trained (run_id={existing.info.run_id})"
            )
            continue

        # Determine Rho (unanimous or mixed)
        rhos = set()
        for rid in run_ids:
            r = client.get_run(rid)
            r_rho = r.data.tags.get("rho")
            if r_rho is not None:
                rhos.add(r_rho)
        rho_sum = rhos.pop() if len(rhos) == 1 else "mixed" if len(rhos) > 1 else None

        # Start run before any artifact loading/training so failures are recorded as FAILED.
        tags = behaviour_tags(
            kind="metalearner",
            behaviour=meta_type,
            component_run_ids=run_ids,
            behaviour_input_hash=bi_hash,
            component_set_hash=cs_hash,
            rho=rho_sum,
            extra={
                "ensemble_name": ens_name,
                "split": split,
                "feature_type": feature_type,
                "similarity_metric": similarity_metric,
                "meta_split_seed": str(meta_split_cfg.seed),
            },
        )

        run = None
        try:
            with mlflow.start_run(
                run_name=f"{ens_name}_adapter_{meta_type}", tags=tags
            ) as run:
                mlflow.log_params(
                    {
                        "adapter_epochs": adapter_epochs,
                        "adapter_lr": adapter_lr,
                        "adapter_batch_size": adapter_batch_size,
                        "meta_split_seed": meta_split_cfg.seed,
                        "meta_split_train": meta_split_cfg.fractions.get("train", 0.6),
                        "meta_split_val": meta_split_cfg.fractions.get("val", 0.2),
                        "adapter_architecture": (
                            "Input -> 128 -> ReLU -> 64 -> ReLU -> num_classes"
                            if meta_type == "meta_mlp"
                            else "Linear"
                        ),
                        "standardization_applied": True,
                    }
                )

                # 3. Secure Feature Extraction
                base_tensors = []
                profile_tensors = []

                for run_id in run_ids:
                    inf_runs = get_inference_run(
                        cfg.mlflow.experiment_name, run_id, split
                    )
                    if inf_runs.empty:
                        raise RuntimeError(f"Missing '{split}' inference for {run_id}")
                    inf_run_id = inf_runs.iloc[0].run_id

                    if use_profiles:
                        prof_runs = get_profile_run(
                            cfg.mlflow.experiment_name, run_id, similarity_metric, split
                        )
                        if prof_runs.empty:
                            raise RuntimeError(
                                f"Missing '{similarity_metric}' profile for {run_id}"
                            )

                        prof_run_id = prof_runs.iloc[0].run_id
                        inf_prof_node = load_mlflow_artifact(
                            prof_run_id,
                            f"profiles/{split}_{similarity_metric}_profiles.pt",
                            file_type="torch",
                            strict=True,
                        ).cpu()

                        # Note: _assert_valid_feature_tensor expects 2D, but profiles are typically 3D [N, Classes]
                        # We bypass the 2D assert here as P is expected to be stacked into 3D.
                        profile_tensors.append(inf_prof_node)

                    data = load_mlflow_artifact(
                        inf_run_id,
                        f"inference_data/{split}_tensors.npz",
                        file_type="numpy",
                        strict=True,
                    )

                    if "logits" in feature_type:
                        if "logits" not in data:
                            raise KeyError(
                                f"Missing logits in inference_data/{split}_tensors.npz for run {inf_run_id}"
                            )
                        base_tensor = torch.from_numpy(data["logits"])
                    elif "embeddings" in feature_type:
                        if "embeddings" not in data:
                            raise KeyError(
                                f"Missing embeddings in inference_data/{split}_tensors.npz for run {inf_run_id}"
                            )
                        base_tensor = torch.from_numpy(data["embeddings"])
                    else:
                        raise ValueError(f"Unknown feature_type: {feature_type}")

                    _assert_valid_feature_tensor(
                        "base_tensor", base_tensor, total_examples
                    )
                    base_tensors.append(base_tensor)

                X_base = torch.cat(base_tensors, dim=1)
                _assert_valid_feature_tensor("X_base", X_base, total_examples)

                if use_profiles:
                    # P shape: (N, M, C)
                    P = torch.stack(profile_tensors, dim=1)
                    N, M, C = P.shape

                    # Dynamically mask out the true class for each example (equivalent to remove_correct_label=True)
                    mask = torch.ones(N, C, dtype=torch.bool, device=P.device)
                    mask[torch.arange(N), labels_tensor] = False

                    # Apply mask and reshape to form profiles purely of off-target predictions
                    mask_3d = mask.unsqueeze(1).expand(N, M, C)
                    P_masked = P[mask_3d].view(N, M, C - 1)

                    # Compute RDM on masked profiles
                    Pc = P_masked - P_masked.mean(dim=2, keepdim=True)
                    P_norm = Pc.norm(dim=2, keepdim=True).clamp_min(1e-8)
                    P_n = Pc / P_norm

                    corr = torch.bmm(P_n, P_n.transpose(1, 2))
                    rdm = 1.0 - corr

                    K = P_n.size(1)
                    idx = torch.triu_indices(K, K, offset=1)
                    S = rdm[:, idx[0], idx[1]]

                    X_all = torch.cat([X_base, S], dim=1)
                else:
                    X_all = X_base

                _assert_valid_feature_tensor("X_all", X_all, total_examples)
                y_all = labels_tensor

                # 4. Construct Data Splits
                train_idx, val_idx, holdout_idx = _three_way_split(
                    total_examples, meta_split_cfg.fractions, meta_split_cfg.seed
                )

                # Split data
                X_train = X_all[train_idx]
                X_val = X_all[val_idx]
                X_holdout = X_all[holdout_idx]

                # Apply Standardization (preventing data leakage)
                X_train, X_val, X_holdout = _standardize_features(
                    X_train, X_val, X_holdout
                )

                ds_train = TensorDataset(X_train, y_all[train_idx])
                ds_val = TensorDataset(X_val, y_all[val_idx])
                ds_hold = TensorDataset(X_holdout, y_all[holdout_idx])

                train_loader = DataLoader(
                    ds_train, batch_size=adapter_batch_size, shuffle=True
                )
                val_loader = DataLoader(
                    ds_val, batch_size=adapter_batch_size, shuffle=False
                )
                hold_loader = DataLoader(
                    ds_hold, batch_size=adapter_batch_size, shuffle=False
                )

                # 5. Build and Train
                num_classes = cfg.dataset.num_classes
                input_dim = X_train.shape[1]

                if meta_type == "meta_lr":
                    model = MetaLR(input_dim, num_classes, bias=adapter_bias)
                elif meta_type == "meta_mlp":
                    model = AdapterMLP(
                        input_dim,
                        num_classes,
                        bias=adapter_bias,
                    )
                else:
                    raise ValueError(f"Unknown meta_type: {meta_type}")

                print(
                    f"  Training {meta_type} (input_dim={input_dim}) for {adapter_epochs} epochs..."
                )

                model, history = _train_adapter(
                    model=model,
                    device=device,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=adapter_epochs,
                    lr=adapter_lr,
                )

                hold_loss, hold_acc, hold_probs = _evaluate_holdout(
                    model, device, hold_loader, nn.CrossEntropyLoss()
                )

                # 6. Logging
                mlflow.log_metrics(
                    {
                        "holdout_loss": hold_loss,
                        "holdout_acc": hold_acc,
                        "val_acc": history[-1]["val_acc"],
                        "train_acc": history[-1]["train_acc"],
                    }
                )

                tabular_path = os.path.join(
                    cache_dir, f"adapter_holdout_{bi_hash}.parquet"
                )
                tensors_path = os.path.join(cache_dir, f"adapter_holdout_{bi_hash}.npz")

                df_hold = pd.DataFrame(
                    {
                        "example_id": [manifest.hashes[i] for i in holdout_idx],
                        "original_index": safe_to_numpy_float64(
                            manifest.original_indices[holdout_idx]
                        ),
                        "label": safe_to_numpy_float64(manifest.labels[holdout_idx]),
                        "prediction": safe_to_numpy_float64(hold_probs.argmax(dim=1)),
                        "confidence": safe_to_numpy_float64(
                            hold_probs.max(dim=1).values
                        ),
                    }
                )

                df_hold.to_parquet(tabular_path, index=False)
                np.savez_compressed(tensors_path, probs=hold_probs.numpy())

                mlflow.log_artifact(tabular_path, artifact_path="adapter_data")
                mlflow.log_artifact(tensors_path, artifact_path="adapter_data")

                log_manifest_lineage(
                    manifest, "train", cfg.dataset.name, context="training"
                )
                log_manifest_lineage(
                    manifest, "val", cfg.dataset.name, context="validation"
                )
                log_manifest_lineage(
                    manifest, "test", cfg.dataset.name, context="testing"
                )

                input_example_np = X_val[:5].numpy()
                model.eval()
                with torch.no_grad():
                    output_example_np = model(X_val[:5].to(device)).cpu().numpy()

                signature = infer_signature(input_example_np, output_example_np)

                mlflow.pytorch.log_model(
                    model,
                    name="model",
                    signature=signature,
                )

                log_resolved_config(cfg)
                print(f"  Holdout Acc = {hold_acc:.4f}  run_id={run.info.run_id}")

        except Exception as e:
            active_run = mlflow.active_run()
            if active_run is not None:
                mlflow.end_run(status="FAILED")
            raise RuntimeError(f"Adapter training failed for '{ens_name}': {e}") from e


if __name__ == "__main__":
    main()
