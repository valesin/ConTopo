#!/usr/bin/env python3
"""
05_train_adapters.py — Meta-learner / adapter training over ensemble components.

This script operates on FINISHED component runs dynamically grouped.
It learns a meta-combination (e.g., Logistic Regression or MLP) over the
frozen representations (logits or embeddings). Topographic profiles can be
appended as additional features.
"""

from __future__ import annotations

import json
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

from src.data.loaders import get_split_labels
from src.ensemble.selector import discover_ensembles_from_cfg
from src.config.paths import get_cache_dir
from src.config.hash import compute_anchor_spec_hash, identity_hash
from src.mlflow_utils import (
    behaviour_tags,
    component_set_hash,
    find_finished_metalearner_run,
    log_resolved_config,
    setup_mlflow,
    get_inference_run,
    get_profile_run,
    load_mlflow_artifact,
    safe_to_numpy_float64,
    log_dataset_lineage,
)

# ─── MODELS ───
from src.networks.heads import LinearAdapter, TwoLayerMLPAdapter, ThreeLayerMLPAdapter
from src.mlflow_schema_logger import (
    log_params as schema_log_params,
    start_run as schema_start_run,
    log_tags as schema_log_tags,
)

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


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_mlflow(cfg)

    split = cfg.execution.split
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
    profile_mask = cfg.adapter.get("profile_mask", "true_class")
    anchor_spec_hash = ""
    if use_profiles:
        anchors_cfg = cfg.profiling.anchors
        anchor_spec_hash = compute_anchor_spec_hash(
            source_split=anchors_cfg.source_split,
            per_class=anchors_cfg.per_class,
            strategy=anchors_cfg.strategy,
            order_by=anchors_cfg.order_by,
            num_classes=cfg.dataset.num_classes,
        )

    # 1. Get ground-truth labels
    labels_tensor = get_split_labels(cfg, split)
    total_examples = len(labels_tensor)

    # 2. Discover Ensemble Components from MLflow
    groups = discover_ensembles_from_cfg(cfg, cfg.mlflow.experiment_name)
    if not groups:
        print("No dynamic ensembles discovered. Exiting.")
        return

    client = mlflow.tracking.MlflowClient()
    dev_name = cfg.runtime.device
    if dev_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev_name)

    for ens_name, run_ids in groups.items():
        print(
            f"\n{'=' * 60}\nAdapters: {ens_name} | Meta: {meta_type} | Feat: {feature_type}"
        )

        cs_hash = component_set_hash(run_ids)
        step_identity_hash = identity_hash(
            "metalearner",
            component_set_hash=cs_hash,
            split=split,
            feature_type=feature_type,
            anchor_spec=anchor_spec_hash,
            meta_split_spec=json.dumps(
                OmegaConf.to_container(meta_split_cfg, resolve=True), sort_keys=True
            ),
            similarity_metric=similarity_metric if use_profiles else "",
            init_seed=str(init_seed),
            profile_mask=profile_mask if use_profiles else "",
            meta_type=meta_type,
        )

        existing = find_finished_metalearner_run(
            cfg.mlflow.experiment_name, step_identity_hash, meta_type=meta_type
        )
        if existing is not None and not cfg.execution.force:
            print(
                f"  SKIP: {meta_type} adapter already trained (run_id={existing.info.run_id})"
            )
            continue

        # Determine Rho (unanimous or mixed)
        rhos = set()
        for rid in run_ids:
            r = client.get_run(rid)
            r_rho = r.data.params.get("rho")
            if r_rho is not None:
                rhos.add(r_rho)
        rho_sum = rhos.pop() if len(rhos) == 1 else "mixed" if len(rhos) > 1 else None

        # Start run before any artifact loading/training so failures are recorded as FAILED.
        tags = behaviour_tags(
            kind="metalearner",
            behaviour=meta_type,
            component_run_ids=run_ids,
            behaviour_input_hash=step_identity_hash,
            component_set_hash=cs_hash,
            rho=rho_sum,
            extra={
                "ensemble_name": ens_name,
                "identity_hash": step_identity_hash,
                "run_name": f"{ens_name}_adapter_{meta_type}",
            },
        )

        run = None
        component_run_ids_csv = ",".join(run_ids)
        tags["component_run_ids_csv"] = component_run_ids_csv
        try:
            with schema_start_run(
                kind="metalearner",
                run_name=f"{ens_name}_adapter_{meta_type}",
                tags=tags,
            ) as run:
                schema_log_params(
                    "metalearner",
                    {
                        "meta_type": meta_type,
                        "feature_type": feature_type,
                        "similarity_metric": (
                            similarity_metric if use_profiles else "none"
                        ),
                        "adapter_epochs": adapter_epochs,
                        "adapter_lr": adapter_lr,
                        "adapter_batch_size": adapter_batch_size,
                        "meta_split_seed": meta_split_cfg.seed,
                        "meta_split_train": meta_split_cfg.fractions.get("train", 0.6),
                        "meta_split_val": meta_split_cfg.fractions.get("val", 0.2),
                        "adapter_architecture": (
                            "TwoLayerMLPAdapter"
                            if meta_type == "meta_mlp_2"
                            else (
                                "ThreeLayerMLPAdapter"
                                if meta_type == "meta_mlp_3"
                                else "LinearAdapter"
                            )
                        ),
                        "standardization_applied": True,
                        "num_components": len(run_ids),
                        "profile_mask": profile_mask if use_profiles else "N/A",
                    },
                )
                schema_log_tags(
                    "metalearner", {"component_run_ids_csv": component_run_ids_csv}
                )

                # 3. Secure Feature Extraction
                base_tensors = []
                profile_tensors = []
                component_logit_preds = []

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

                        profile_tensors.append(inf_prof_node)

                    data = load_mlflow_artifact(
                        inf_run_id,
                        f"inference_data/{split}_tensors.npz",
                        file_type="numpy",
                        strict=True,
                    )

                    if "logits" in data:
                        component_logit_preds.append(
                            torch.from_numpy(data["logits"]).argmax(dim=1)
                        )
                    else:
                        component_logit_preds.append(None)

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

                    if profile_mask == "true_class":
                        # Dynamically mask out the true class for each example
                        mask = torch.ones(N, C, dtype=torch.bool, device=P.device)
                        mask[torch.arange(N), labels_tensor] = False
                        mask_3d = mask.unsqueeze(1).expand(N, M, C)
                        P_masked = P[mask_3d].view(N, M, C - 1)
                    elif profile_mask == "argmax_similarity":
                        preds = P.argmax(dim=2)
                        mask = torch.ones(N, M, C, dtype=torch.bool, device=P.device)
                        mask[
                            torch.arange(N).unsqueeze(1),
                            torch.arange(M).unsqueeze(0),
                            preds,
                        ] = False
                        P_masked = P[mask].view(N, M, C - 1)
                    elif profile_mask == "argmax_logits":
                        if any(p is None for p in component_logit_preds):
                            raise ValueError(
                                "Cannot use argmax_logits profile mask because not all components have cached logits."
                            )
                        preds = torch.stack(component_logit_preds, dim=1).to(P.device)
                        mask = torch.ones(N, M, C, dtype=torch.bool, device=P.device)
                        mask[
                            torch.arange(N).unsqueeze(1),
                            torch.arange(M).unsqueeze(0),
                            preds,
                        ] = False
                        P_masked = P[mask].view(N, M, C - 1)
                    elif profile_mask == "none":
                        P_masked = P
                    else:
                        raise ValueError(f"Unknown profile_mask: {profile_mask}")

                    # Compute RDM on masked profiles
                    Pc = P_masked - P_masked.mean(dim=2, keepdim=True)
                    P_norm = Pc.norm(dim=2, keepdim=True).clamp_min(1e-8)
                    P_n = Pc / P_norm

                    corr = torch.bmm(P_n, P_n.transpose(1, 2))
                    rdm = 1.0 - corr

                    K = P_n.size(1)
                    idx = torch.triu_indices(K, K, offset=1)
                    S = rdm[:, idx[0], idx[1]]
                    profile_feature_dim = int(S.shape[1])

                    X_all = torch.cat([X_base, S], dim=1)
                else:
                    profile_feature_dim = 0
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

                # Capture standardization stats for traceability
                standardize_mean = X_train.mean(dim=0, keepdim=True)
                standardize_std = X_train.std(dim=0, keepdim=True)

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
                    model = LinearAdapter(input_dim, num_classes, bias=adapter_bias)
                elif meta_type == "meta_mlp_2":
                    model = TwoLayerMLPAdapter(
                        in_dim=input_dim,
                        hidden_dim=128,
                        num_classes=num_classes,
                        dropout=0.0,
                        bias=adapter_bias,
                    )
                elif meta_type == "meta_mlp_3":
                    model = ThreeLayerMLPAdapter(
                        in_dim=input_dim,
                        num_classes=num_classes,
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
                    cache_dir, f"adapter_holdout_{step_identity_hash}.parquet"
                )
                tensors_path = os.path.join(
                    cache_dir, f"adapter_holdout_{step_identity_hash}.npz"
                )
                inputs_path = os.path.join(
                    cache_dir, f"adapter_inputs_{step_identity_hash}.npz"
                )
                split_trace_path = os.path.join(
                    cache_dir, f"adapter_split_trace_{step_identity_hash}.parquet"
                )

                os.makedirs(cache_dir, exist_ok=True)

                # Save exact regression-head inputs used in this run
                np.savez_compressed(
                    inputs_path,
                    X_train=X_train.detach().cpu().numpy(),
                    X_val=X_val.detach().cpu().numpy(),
                    X_holdout=X_holdout.detach().cpu().numpy(),
                    y_train=y_all[train_idx].detach().cpu().numpy(),
                    y_val=y_all[val_idx].detach().cpu().numpy(),
                    y_holdout=y_all[holdout_idx].detach().cpu().numpy(),
                    train_idx=np.asarray(train_idx),
                    val_idx=np.asarray(val_idx),
                    holdout_idx=np.asarray(holdout_idx),
                    standardize_mean=standardize_mean.detach().cpu().numpy(),
                    standardize_std=standardize_std.detach().cpu().numpy(),
                    x_base_dim=np.asarray([int(X_base.shape[1])], dtype=np.int64),
                    profile_feature_dim=np.asarray(
                        [profile_feature_dim], dtype=np.int64
                    ),
                    use_profiles=np.asarray([int(use_profiles)], dtype=np.int64),
                    profile_mask=np.asarray([profile_mask]),
                )

                split_trace = pd.DataFrame(
                    {
                        "original_index": np.concatenate(
                            [train_idx, val_idx, holdout_idx]
                        ),
                        "split": (
                            ["train"] * len(train_idx)
                            + ["val"] * len(val_idx)
                            + ["holdout"] * len(holdout_idx)
                        ),
                        "position_in_split": np.concatenate(
                            [
                                np.arange(len(train_idx)),
                                np.arange(len(val_idx)),
                                np.arange(len(holdout_idx)),
                            ]
                        ),
                    }
                )
                split_trace.to_parquet(split_trace_path, index=False)

                df_hold = pd.DataFrame(
                    {
                        "original_index": safe_to_numpy_float64(
                            torch.tensor(holdout_idx)
                        ),
                        "label": safe_to_numpy_float64(labels_tensor[holdout_idx]),
                        "prediction": safe_to_numpy_float64(hold_probs.argmax(dim=1)),
                        "confidence": safe_to_numpy_float64(
                            hold_probs.max(dim=1).values
                        ),
                    }
                )

                df_hold.to_parquet(tabular_path, index=False)
                np.savez_compressed(tensors_path, probs=hold_probs.numpy())

                mlflow.log_artifact(inputs_path, artifact_path="adapter_inputs")
                mlflow.log_artifact(split_trace_path, artifact_path="adapter_inputs")
                mlflow.log_artifact(tabular_path, artifact_path="adapter_data")
                mlflow.log_artifact(tensors_path, artifact_path="adapter_data")

                log_dataset_lineage(
                    y_all[train_idx],
                    f"meta_train_from_{split}",
                    cfg.dataset.name,
                    context="training",
                )
                log_dataset_lineage(
                    y_all[val_idx],
                    f"meta_val_from_{split}",
                    cfg.dataset.name,
                    context="validation",
                )
                log_dataset_lineage(
                    y_all[holdout_idx],
                    f"meta_holdout_from_{split}",
                    cfg.dataset.name,
                    context="testing",
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
