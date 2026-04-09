#!/usr/bin/env python3
"""
05_train_adapters.py — Meta-learner / adapter training over ensemble components.

This script operates on FINISHED component runs dynamically grouped.
It learns a meta-combination (e.g., Logistic Regression or MLP) over the
frozen representations (logits or embeddings). Topographic profiles can be
appended as additional features.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import warnings
import numpy as np
from numpy.exceptions import VisibleDeprecationWarning
import pandas as pd

import hydra

# Suppress torchvision's NumPy 2.4 deprecation warning in CIFAR dataset loading
warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)
import mlflow
import torch.backends.cudnn as cudnn
from mlflow.models.signature import infer_signature
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from src.data.loaders import get_num_classes, get_split_labels
from src.ensemble.selector import discover_ensembles_from_cfg
from src.config.hash import compute_anchor_spec_hash, identity_hash
from src.mlflow_utils import (
    behaviour_tags,
    component_set_hash,
    find_finished_metalearner_run,
    log_resolved_config,
    setup_mlflow,
    find_finished_identity_run,
    load_mlflow_artifact,
    set_torch_seed,
    resolve_device,
    get_run_context,
    safe_to_numpy_float64,
    log_dataset_lineage,
)

# ─── MODELS ───
from src.networks.heads import (
    LinearAdapter,
    TwoLayerMLPAdapter,
    ThreeLayerMLPAdapter,
    FourLayerMLPAdapter,
)
from src.mlflow_schema_logger import (
    log_params as schema_log_params,
    start_run as schema_start_run,
    log_tags as schema_log_tags,
    timed_log_metrics,
    timed_log_artifact,
    timed_log_model,
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


# ─── HYBRID PROFILE MASKS ───

_HYBRID_MASKS = {
    "hybrid_trueclass_argmaxlogits": ("true_class", "argmax_logits"),
    "hybrid_trueclass_argmaxsimilarity": ("true_class", "argmax_similarity"),
}


def _apply_profile_mask(
    P: torch.Tensor,
    mask_type: str,
    labels: torch.Tensor,
    component_logit_preds: list[torch.Tensor | None],
    indices: np.ndarray | None = None,
) -> torch.Tensor:
    """Apply a single profile mask to P (N, M, C) → (N, M, C-1) or (N, M, C).

    Removes one class dimension per sample before RDM computation so that the
    resulting features reflect inter-model agreement structure rather than
    raw class affinity.

    Modes:
      true_class          - removes the ground-truth class (requires labels;
                            causes label leakage if applied to holdout).
      argmax_similarity   - removes the class with highest mean similarity across
                            models (label-free, safe at inference time).
      argmax_logits       - removes the class with highest mean ensemble logit
                            (label-free, safe at inference time).
      none                - no masking; returns P unchanged.

    Hybrid modes (defined in _HYBRID_MASKS) apply true_class to train/val and
    an argmax variant to holdout, eliminating leakage at evaluation while
    preserving a stronger training signal.
    """
    N, M, C = P.shape

    if not torch.isfinite(P).all():
        raise ValueError("Profile tensor P contains NaN/Inf values")

    if mask_type == "true_class":
        mask = torch.ones(N, C, dtype=torch.bool, device=P.device)
        mask[torch.arange(N), labels] = False
        mask_3d = mask.unsqueeze(1).expand(N, M, C)
        return P[mask_3d].view(N, M, C - 1)

    elif mask_type == "argmax_similarity":
        mean_similarity = P.mean(dim=1)
        preds = mean_similarity.argmax(dim=1)
        mask = torch.ones(N, M, C, dtype=torch.bool, device=P.device)
        mask[
            torch.arange(N).unsqueeze(1),
            torch.arange(M).unsqueeze(0),
            preds.unsqueeze(1),
        ] = False
        return P[mask].view(N, M, C - 1)

    elif mask_type == "argmax_logits":
        if any(p is None for p in component_logit_preds):
            raise ValueError(
                "Cannot use argmax_logits profile mask because not all components have cached logits."
            )

        if any(
            not torch.isfinite(p).all() for p in component_logit_preds if p is not None
        ):
            raise ValueError(
                "Cannot use argmax_logits profile mask because logits contain NaN/Inf values."
            )

        if indices is not None:
            mean_logits = (
                torch.stack([p[indices] for p in component_logit_preds], dim=1)
                .to(P.device)
                .mean(dim=1)
            )
        else:
            mean_logits = (
                torch.stack(component_logit_preds, dim=1).to(P.device).mean(dim=1)
            )
        preds = mean_logits.argmax(dim=1)
        mask = torch.ones(N, M, C, dtype=torch.bool, device=P.device)
        mask[
            torch.arange(N).unsqueeze(1),
            torch.arange(M).unsqueeze(0),
            preds.unsqueeze(1),
        ] = False
        return P[mask].view(N, M, C - 1)

    elif mask_type == "none":
        return P

    else:
        raise ValueError(f"Unknown profile_mask: {mask_type}")


def _compute_rdm_features(P_masked: torch.Tensor) -> torch.Tensor:
    """Compute upper-triangular RDM features from masked profiles.

    Args:
        P_masked: (N, M, C') tensor of masked profiles
    Returns:
        S: (N, K*(K-1)/2) tensor of pairwise dissimilarities
    """
    Pc = P_masked - P_masked.mean(dim=2, keepdim=True)
    P_norm = Pc.norm(dim=2, keepdim=True).clamp_min(1e-8)
    P_n = Pc / P_norm

    corr = torch.bmm(P_n, P_n.transpose(1, 2))
    rdm = 1.0 - corr

    K = P_n.size(1)
    idx = torch.triu_indices(K, K, offset=1)
    return rdm[:, idx[0], idx[1]]


# ─── MAIN ───


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_mlflow(cfg)

    split = cfg.execution.split

    init_seed = cfg.adapter.init_seed

    # Adapter training is small but must be fully reproducible: each ensemble's
    # adapter starts from the same fixed seed regardless of iteration order.
    # cuDNN determinism is enforced here (benchmark off) so weight-init and
    # DataLoader shuffles produce identical results across reruns.
    cudnn.deterministic = True
    cudnn.benchmark = False

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
            num_classes=get_num_classes(cfg.dataset.name),
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
    device = resolve_device(cfg.runtime.device)

    for ens_name, run_ids in groups.items():
        # Re-seed at the start of every ensemble so each adapter's weight
        # initialisation and DataLoader shuffles are independent of the order
        # in which other ensembles were processed.
        set_torch_seed(init_seed)
        profile_info = (
            f" | Sim: {similarity_metric} | Mask: {profile_mask}"
            if use_profiles
            else ""
        )
        print(
            f"\n{'=' * 60}\n"
            f"Adapters: {ens_name} | Meta: {meta_type} | Feat: {feature_type}"
            f"{profile_info} | Seed: {init_seed}"
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
            r_rho, _, _ = get_run_context(r)
            if r_rho != "?":
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
                                else (
                                    "FourLayerMLPAdapter"
                                    if meta_type == "meta_mlp_4"
                                    else "LinearAdapter"
                                )
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
                    inf_identity = identity_hash(
                        "inference", trained_model_run_id=run_id, split=split
                    )
                    inf_run = find_finished_identity_run(
                        cfg.mlflow.experiment_name, "inference", inf_identity
                    )
                    if inf_run is None:
                        raise RuntimeError(f"Missing '{split}' inference for {run_id}")
                    inf_run_id = inf_run.info.run_id

                    if use_profiles:
                        prof_identity = identity_hash(
                            "category_similarity_profile",
                            parent_run_id=run_id,
                            anchor_spec_hash=anchor_spec_hash,
                            similarity_metric=similarity_metric,
                            split=split,
                        )
                        prof_run = find_finished_identity_run(
                            cfg.mlflow.experiment_name,
                            "category_similarity_profile",
                            prof_identity,
                        )
                        if prof_run is None:
                            raise RuntimeError(
                                f"Missing '{similarity_metric}' profile for {run_id}"
                            )
                        prof_run_id = prof_run.info.run_id
                        inf_prof_node = load_mlflow_artifact(
                            prof_run_id,
                            f"profiles/{split}_{similarity_metric}_profiles.pt",
                            file_type="torch",
                            strict=True,
                            cache_dir=cfg.mlflow.artifact_cache_dir,
                        ).cpu()

                        if not torch.isfinite(inf_prof_node).all():
                            raise ValueError(
                                f"Profile artifact for model {run_id} (profile run {prof_run_id}) "
                                f"contains NaN/Inf — likely corrupted artifact on disk"
                            )

                        profile_tensors.append(inf_prof_node)

                    data = load_mlflow_artifact(
                        inf_run_id,
                        f"inference/{split}_tensors.npz",
                        file_type="numpy",
                        strict=True,
                        cache_dir=cfg.mlflow.artifact_cache_dir,
                    )

                    if "logits" in data:
                        component_logit_preds.append(torch.from_numpy(data["logits"]))
                    else:
                        component_logit_preds.append(None)

                    if "logits" in feature_type:
                        if "logits" not in data:
                            raise KeyError(
                                f"Missing logits in inference/{split}_tensors.npz for run {inf_run_id}"
                            )
                        base_tensor = torch.from_numpy(data["logits"])
                    elif "embeddings" in feature_type:
                        if "embeddings" not in data:
                            raise KeyError(
                                f"Missing embeddings in inference/{split}_tensors.npz for run {inf_run_id}"
                            )
                        base_tensor = torch.nn.functional.normalize(
                            torch.from_numpy(data["embeddings"]).float(), p=2, dim=1
                        )
                    else:
                        raise ValueError(f"Unknown feature_type: {feature_type}")

                    _assert_valid_feature_tensor(
                        "base_tensor", base_tensor, total_examples
                    )
                    base_tensors.append(base_tensor)

                X_base = torch.cat(base_tensors, dim=1)
                _assert_valid_feature_tensor("X_base", X_base, total_examples)

                # 4. Construct Data Splits (needed before masking for hybrid modes)
                train_idx, val_idx, holdout_idx = _three_way_split(
                    total_examples, meta_split_cfg.fractions, meta_split_cfg.seed
                )
                y_all = labels_tensor

                if use_profiles:
                    # P shape: (N, M, C)
                    P = torch.stack(profile_tensors, dim=1)

                    if profile_mask in _HYBRID_MASKS:
                        # Hybrid: true_class on train/val, argmax on holdout
                        tv_mask, ho_mask = _HYBRID_MASKS[profile_mask]

                        P_train = _apply_profile_mask(
                            P[train_idx],
                            tv_mask,
                            labels_tensor[train_idx],
                            component_logit_preds,
                            train_idx,
                        )
                        P_val = _apply_profile_mask(
                            P[val_idx],
                            tv_mask,
                            labels_tensor[val_idx],
                            component_logit_preds,
                            val_idx,
                        )
                        P_hold = _apply_profile_mask(
                            P[holdout_idx],
                            ho_mask,
                            labels_tensor[holdout_idx],
                            component_logit_preds,
                            holdout_idx,
                        )

                        S_train = _compute_rdm_features(P_train)
                        S_val = _compute_rdm_features(P_val)
                        S_hold = _compute_rdm_features(P_hold)
                        profile_feature_dim = int(S_train.shape[1])

                        X_train = torch.cat([X_base[train_idx], S_train], dim=1)
                        X_val = torch.cat([X_base[val_idx], S_val], dim=1)
                        X_holdout = torch.cat([X_base[holdout_idx], S_hold], dim=1)
                    else:
                        # Uniform mask on full dataset
                        P_masked = _apply_profile_mask(
                            P,
                            profile_mask,
                            labels_tensor,
                            component_logit_preds,
                        )
                        S = _compute_rdm_features(P_masked)
                        profile_feature_dim = int(S.shape[1])

                        X_all = torch.cat([X_base, S], dim=1)
                        _assert_valid_feature_tensor("X_all", X_all, total_examples)
                        X_train = X_all[train_idx]
                        X_val = X_all[val_idx]
                        X_holdout = X_all[holdout_idx]
                else:
                    profile_feature_dim = 0
                    X_train = X_base[train_idx]
                    X_val = X_base[val_idx]
                    X_holdout = X_base[holdout_idx]

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
                num_classes = get_num_classes(cfg.dataset.name)
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
                elif meta_type == "meta_mlp_4":
                    model = FourLayerMLPAdapter(
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
                timed_log_metrics(
                    {
                        "holdout_loss": hold_loss,
                        "holdout_acc": hold_acc,
                        "val_acc": history[-1]["val_acc"],
                        "train_acc": history[-1]["train_acc"],
                    }
                )

                with tempfile.TemporaryDirectory() as tmpdir:
                    tabular_path = os.path.join(
                        tmpdir, f"adapter_holdout_{step_identity_hash}.parquet"
                    )
                    tensors_path = os.path.join(
                        tmpdir, f"adapter_holdout_{step_identity_hash}.npz"
                    )
                    inputs_path = os.path.join(
                        tmpdir, f"adapter_inputs_{step_identity_hash}.npz"
                    )
                    split_trace_path = os.path.join(
                        tmpdir, f"adapter_split_trace_{step_identity_hash}.parquet"
                    )

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
                            "prediction": safe_to_numpy_float64(
                                hold_probs.argmax(dim=1)
                            ),
                            "confidence": safe_to_numpy_float64(
                                hold_probs.max(dim=1).values
                            ),
                        }
                    )

                    df_hold.to_parquet(tabular_path, index=False)
                    np.savez_compressed(tensors_path, probs=hold_probs.numpy())

                    timed_log_artifact(inputs_path, artifact_path="inputs")
                    timed_log_artifact(split_trace_path, artifact_path="inputs")
                    timed_log_artifact(tabular_path, artifact_path="data")
                    timed_log_artifact(tensors_path, artifact_path="data")

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

                timed_log_model(
                    model,
                    name="model",
                    signature=signature,
                )

                log_resolved_config(cfg)
                print(f"  Holdout Acc = {hold_acc:.4f}  run_id={run.info.run_id}")

        except Exception as e:
            # The `with` context manager already called end_run(status="FINISHED"),
            # so mlflow.active_run() is None here. Use the client to correct the status.
            if run is not None:
                client.set_terminated(run.info.run_id, status="FAILED")
            raise RuntimeError(f"Adapter training failed for '{ens_name}': {e}") from e


if __name__ == "__main__":
    main()
