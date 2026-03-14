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


class MetaMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.3,
        bias: bool = True,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=bias),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes, bias=bias),
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
    adapter_dropout = cfg.adapter.dropout

    meta_type = cfg.adapter.meta_type
    feature_type = cfg.adapter.feature_type
    similarity_metric = cfg.adapter.similarity_metric
    hidden_dim = cfg.adapter.hidden_dim
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

        try:
            # 3. Secure Feature Extraction
            base_tensors = []
            profile_tensors = []

            for run_id in run_ids:
                # Get inference base
                inf_runs = get_inference_run(cfg.mlflow.experiment_name, run_id, split)
                if inf_runs.empty:
                    raise RuntimeError(f"Missing '{split}' inference for {run_id}")
                inf_run_id = inf_runs.iloc[0].run_id

                # Optional Profile Extraction
                inf_prof_node = None
                if use_profiles:
                    prof_runs = get_profile_run(cfg.mlflow.experiment_name, run_id, similarity_metric, split)
                    if prof_runs.empty:
                        raise RuntimeError(f"Missing '{similarity_metric}' profile for {run_id}")

                    prof_run_id = prof_runs.iloc[0].run_id
                    inf_prof_node = load_mlflow_artifact(prof_run_id, f"profiles/{split}_{similarity_metric}_profiles.pt", file_type="torch", strict=True).cpu()
                    profile_tensors.append(inf_prof_node)

                # Get Base Tensor
                data = load_mlflow_artifact(inf_run_id, f"inference_data/{split}_tensors.npz", file_type="numpy", strict=True)

                base_tensor = None
                if "logits" in feature_type:
                    base_tensor = torch.from_numpy(data["logits"])
                elif "embeddings" in feature_type:
                    base_tensor = torch.from_numpy(data["embeddings"])
                else:
                    raise ValueError(f"Unknown feature_type: {feature_type}")

                base_tensors.append(base_tensor)

            # Assemble Concatenated Base Features [N, K * dim]
            X_base = torch.cat(base_tensors, dim=1)
            
            if use_profiles:
                # P shape: [N, K, num_classes]
                P = torch.stack(profile_tensors, dim=1)
                
                # Pearson correlation across the num_classes dim (dim=2)
                Pc = P - P.mean(dim=2, keepdim=True)
                P_norm = Pc.norm(dim=2, keepdim=True).clamp_min(1e-8)
                P_n = Pc / P_norm  # [N, K, num_classes]
                
                # Correlation distance = 1 - pearson correlation
                corr = torch.bmm(P_n, P_n.transpose(1, 2))  # [N, K, K]
                rdm = 1.0 - corr                            # [N, K, K]
                
                # Extract upper triangle without the diagonal
                K = P_n.size(1)
                idx = torch.triu_indices(K, K, offset=1)
                S = rdm[:, idx[0], idx[1]]  # [N, K*(K-1)/2]
                
                # Assemble Feature Matrix [N, K*dim + K*(K-1)/2]
                X_all = torch.cat([X_base, S], dim=1)
            else:
                X_all = X_base

            y_all = labels_tensor

            # 4. Construct Data Splits
            train_idx, val_idx, holdout_idx = _three_way_split(
                total_examples, meta_split_cfg.fractions, meta_split_cfg.seed
            )

            ds_train = TensorDataset(X_all[train_idx], y_all[train_idx])
            ds_val = TensorDataset(X_all[val_idx], y_all[val_idx])
            ds_hold = TensorDataset(X_all[holdout_idx], y_all[holdout_idx])

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
            input_dim = X_all.shape[1]

            if meta_type == "meta_lr":
                model = MetaLR(input_dim, num_classes, bias=adapter_bias)
            elif meta_type == "meta_mlp":
                model = MetaMLP(
                    input_dim, hidden_dim, num_classes, adapter_dropout, adapter_bias
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

            # Determine Rho (unanimous or mixed)
            rhos = set()
            for rid in run_ids:
                r = client.get_run(rid)
                r_rho = r.data.tags.get("rho")
                if r_rho is not None:
                    rhos.add(r_rho)
            rho_sum = rhos.pop() if len(rhos) == 1 else "mixed" if len(rhos) > 1 else None

            # 6. Logging
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

            with mlflow.start_run(
                run_name=f"{ens_name}_adapter_{meta_type}", tags=tags
            ) as run:
                mlflow.log_params(
                    {
                        "adapter_epochs": adapter_epochs,
                        "adapter_lr": adapter_lr,
                        "adapter_batch_size": adapter_batch_size,
                        "adapter_dropout": adapter_dropout,
                        "hidden_dim": hidden_dim if meta_type == "meta_mlp" else None,
                        "meta_split_seed": meta_split_cfg.seed,
                        "meta_split_train": meta_split_cfg.fractions.get("train", 0.6),
                        "meta_split_val": meta_split_cfg.fractions.get("val", 0.2),
                    }
                )

                # Log final holdout metrics directly
                mlflow.log_metrics(
                    {
                        "holdout_loss": hold_loss,
                        "holdout_acc": hold_acc,
                        "val_acc": history[-1]["val_acc"],
                        "train_acc": history[-1]["train_acc"],
                    }
                )

                # Parquet / NPZ Logging (Same structure as 04_run_ensemble natively scoped just for holdout subset)
                tabular_path = os.path.join(
                    cache_dir, f"adapter_holdout_{bi_hash}.parquet"
                )
                tensors_path = os.path.join(cache_dir, f"adapter_holdout_{bi_hash}.npz")

                df_hold = pd.DataFrame(
                    {
                        "example_id": [manifest.hashes[i] for i in holdout_idx],
                        "original_index": safe_to_numpy_float64(manifest.original_indices[holdout_idx]),
                        "label": safe_to_numpy_float64(manifest.labels[holdout_idx]),
                        "prediction": safe_to_numpy_float64(hold_probs.argmax(dim=1)),
                        "confidence": safe_to_numpy_float64(hold_probs.max(dim=1).values),
                    }
                )

                df_hold.to_parquet(tabular_path, index=False)
                np.savez_compressed(tensors_path, probs=hold_probs.numpy())

                mlflow.log_artifact(tabular_path, artifact_path="adapter_data")
                mlflow.log_artifact(tensors_path, artifact_path="adapter_data")

                # Dataset Schema Tracking
                # Use standard manifest logging
                log_manifest_lineage(manifest, "train", cfg.dataset.name, context="training")
                log_manifest_lineage(manifest, "val", cfg.dataset.name, context="validation")
                log_manifest_lineage(manifest, "test", cfg.dataset.name, context="testing")

                # Infer signature and log the adapter model
                input_example_np = X_all[val_idx][:5].numpy()
                model.eval()
                with torch.no_grad():
                    output_example_np = model(X_all[val_idx][:5].to(device)).cpu().numpy()
                
                signature = infer_signature(input_example_np, output_example_np)
                
                mlflow.pytorch.log_model(
                    model,
                    name="model",
                    signature=signature,
                )

                log_resolved_config(cfg)
                print(f"  Holdout Acc = {hold_acc:.4f}  run_id={run.info.run_id}")

        except Exception as e:
            print(f"  FAIL: {e}")


if __name__ == "__main__":
    main()
