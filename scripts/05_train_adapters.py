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
from src.ensemble.selector import discover_ensembles_with_runs_from_cfg
from src.config.hash import compute_anchor_spec_hash, identity_hash
from src.mlflow_utils import (
    apply_mlflow_env_overrides,
    behaviour_tags,
    component_set_hash,
    log_resolved_config,
    setup_mlflow,
    set_torch_seed,
    resolve_device,
    get_run_context,
    safe_to_numpy_float64,
    log_dataset_lineage,
)
from src.repositories.functional_run_repository import (
    configure_run_repository,
    find_finished_identity_run,
)

from src.mlflow_schema_logger import (
    log_params as schema_log_params,
    start_run as schema_start_run,
    log_tags as schema_log_tags,
    timed_log_metrics,
    timed_log_artifact,
    timed_log_model,
)

# ─── EXTRACTED COMPONENTS ───
from src.profiling.masking import (
    HYBRID_MASKS,
    apply_profile_mask,
    compute_rdm_features,
    assert_valid_feature_tensor,
)
from src.networks.adapter_registry import build_adapter, adapter_architecture_name
from src.adapter.feature_extraction import extract_component_features
from src.training.adapter_training import (
    three_way_split,
    standardize_features,
    train_adapter,
    evaluate_holdout,
)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    apply_mlflow_env_overrides(cfg)
    setup_mlflow(cfg)
    configure_run_repository(cfg.mlflow.tracking_uri, cfg.mlflow.experiment_name)

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
    groups, run_index = discover_ensembles_with_runs_from_cfg(
        cfg, cfg.mlflow.experiment_name
    )
    if not groups:
        print("No dynamic ensembles discovered. Exiting.")
        return

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

        existing = find_finished_identity_run("metalearner", step_identity_hash)
        if existing is not None and not cfg.execution.force:
            print(
                f"  SKIP: {meta_type} adapter already trained (run_id={existing.info.run_id})"
            )
            continue

        # Determine Rho (unanimous or mixed)
        rhos = set()
        for rid in run_ids:
            r = run_index[rid]
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
                        "adapter_architecture": adapter_architecture_name(meta_type),
                        "standardization_applied": True,
                        "num_components": len(run_ids),
                        "profile_mask": profile_mask if use_profiles else "N/A",
                    },
                )
                schema_log_tags(
                    "metalearner", {"component_run_ids_csv": component_run_ids_csv}
                )

                # 3. Secure Feature Extraction
                features = extract_component_features(
                    run_ids=run_ids,
                    split=split,
                    feature_type=feature_type,
                    use_profiles=use_profiles,
                    similarity_metric=similarity_metric,
                    anchor_spec_hash=anchor_spec_hash,
                    cfg=cfg,
                    total_examples=total_examples,
                )
                base_tensors = features.base_tensors
                profile_tensors = features.profile_tensors
                component_logit_preds = features.component_logit_preds

                X_base = torch.cat(base_tensors, dim=1)
                assert_valid_feature_tensor("X_base", X_base, total_examples)

                # 4. Construct Data Splits (needed before masking for hybrid modes)
                train_idx, val_idx, holdout_idx = three_way_split(
                    total_examples, meta_split_cfg.fractions, meta_split_cfg.seed
                )
                y_all = labels_tensor

                if use_profiles:
                    # P shape: (N, M, C)
                    P = torch.stack(profile_tensors, dim=1)

                    if profile_mask in HYBRID_MASKS:
                        # Hybrid: true_class on train/val, argmax on holdout
                        tv_mask, ho_mask = HYBRID_MASKS[profile_mask]

                        P_train = apply_profile_mask(
                            P[train_idx],
                            tv_mask,
                            labels_tensor[train_idx],
                            component_logit_preds,
                            train_idx,
                        )
                        P_val = apply_profile_mask(
                            P[val_idx],
                            tv_mask,
                            labels_tensor[val_idx],
                            component_logit_preds,
                            val_idx,
                        )
                        P_hold = apply_profile_mask(
                            P[holdout_idx],
                            ho_mask,
                            labels_tensor[holdout_idx],
                            component_logit_preds,
                            holdout_idx,
                        )

                        S_train = compute_rdm_features(P_train)
                        S_val = compute_rdm_features(P_val)
                        S_hold = compute_rdm_features(P_hold)
                        profile_feature_dim = int(S_train.shape[1])

                        X_train = torch.cat([X_base[train_idx], S_train], dim=1)
                        X_val = torch.cat([X_base[val_idx], S_val], dim=1)
                        X_holdout = torch.cat([X_base[holdout_idx], S_hold], dim=1)
                    else:
                        # Uniform mask on full dataset
                        P_masked = apply_profile_mask(
                            P,
                            profile_mask,
                            labels_tensor,
                            component_logit_preds,
                        )
                        S = compute_rdm_features(P_masked)
                        profile_feature_dim = int(S.shape[1])

                        X_all = torch.cat([X_base, S], dim=1)
                        assert_valid_feature_tensor("X_all", X_all, total_examples)
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
                X_train, X_val, X_holdout = standardize_features(
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

                model = build_adapter(
                    meta_type=meta_type,
                    input_dim=input_dim,
                    num_classes=num_classes,
                    bias=adapter_bias,
                )

                print(
                    f"  Training {meta_type} (input_dim={input_dim}) for {adapter_epochs} epochs..."
                )

                model, history = train_adapter(
                    model=model,
                    device=device,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=adapter_epochs,
                    lr=adapter_lr,
                )

                hold_loss, hold_acc, hold_probs = evaluate_holdout(
                    model, device, hold_loader, nn.CrossEntropyLoss()
                )

                # 6. Logging
                timed_log_metrics(
                    {
                        "holdout_loss": hold_loss,
                        "holdout_acc": hold_acc,
                        "val_acc": history[-1]["val_acc"],
                        "val_loss": history[-1]["val_loss"],
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
                mlflow.tracking.MlflowClient().set_terminated(
                    run.info.run_id, status="FAILED"
                )
            raise RuntimeError(f"Adapter training failed for '{ens_name}': {e}") from e


if __name__ == "__main__":
    main()
