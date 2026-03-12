#!/usr/bin/env python3
"""
05_train_adapters.py — Hydra + MLflow meta-learner adapter training.

For each ensemble, trains the meta-learner specified by ``cfg.adapter.*``
using a THREE-WAY SPLIT (train/val/holdout) of the evaluation data.
Logs exact example_id membership for each split partition.

Meta-learner identity is now driven by Hydra adapter config:
  - ``adapter.meta_type``:        meta_lr | meta_mlp
  - ``adapter.feature_type``:     logits | embeddings | embeddings+profiles
  - ``adapter.similarity_metric``: cosine | l2 (used with embeddings/profiles)
  - ``adapter.hidden_dim``:       MLP hidden dim (meta_mlp only)

Anchor selection is configured via ``adapter.anchor_selection``.

Usage:
    # Single run:
    python scripts/05_train_adapters.py

    # Sweep meta-learner types and feature types:
    python scripts/05_train_adapters.py --multirun \\
        adapter.meta_type=meta_lr,meta_mlp \\
        adapter.feature_type=logits,embeddings,embeddings+profiles

    # Also sweep bias:
    python scripts/05_train_adapters.py --multirun \\
        adapter.meta_type=meta_lr,meta_mlp \\
        adapter.feature_type=logits,embeddings \\
        adapter.bias=true,false
"""

from __future__ import annotations

import json
import os
import tempfile

import hydra
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf

from src.ensemble.selector import resolve_components
from src.data.cache import get_backend
from src.data.anchors import get_or_create_anchors, anchor_spec_hash
from src.data.manifest import get_or_create_manifest
from src.networks.heads import LinearAdapter, ThreeLayerMLPAdapter
from src.profiling.category_similarity import (
    compute_similarity_profile,
    similarity_profile_hash,
)
from src.mlflow_utils import (
    behavior_tags,
    category_similarity_profile_tags,
    component_set_hash,
    behavior_input_hash,
    find_finished_behavior_run,
    find_finished_similarity_profile_run,
    log_git_info,
    log_resolved_config,
    setup_mlflow,
)


def _resolve_anchor_selection(cfg):
    """Resolve anchor_selection from adapter config."""
    anchor_sel = OmegaConf.to_container(cfg.adapter.anchor_selection, resolve=True)
    return anchor_sel


def _load_features_for_runs(run_ids, artifacts_root, split="test", feature_key="logits"):
    """Load a single feature type (logits or embeddings) for all runs → [N, M*D].

    HARD FAIL on missing artifacts.
    """
    backend = get_backend("pt")
    features_list = []
    labels = None

    for run_id in run_ids:
        artifact_dir = os.path.join(artifacts_root, "inference", run_id, split)
        feat_path = os.path.join(artifact_dir, f"{feature_key}{backend.extension}")

        if not backend.exists(feat_path):
            raise FileNotFoundError(
                f"HARD FAIL: {feature_key} not found at {feat_path}. "
                f"Run scripts/02_cache_inference.py first."
            )
        features_list.append(backend.load(feat_path))

        if labels is None:
            labels_path = os.path.join(artifact_dir, f"labels{backend.extension}")
            if backend.exists(labels_path):
                labels = backend.load(labels_path)

    if labels is None:
        raise FileNotFoundError("HARD FAIL: could not find labels for any component run.")

    concat = torch.cat(features_list, dim=1)  # [N, M*D]
    return concat, labels


def _get_or_compute_similarity_profile(
    *,
    run_id: str,
    anchors: dict,
    similarity_metric: str,
    split: str,
    artifacts_root: str,
    experiment_name: str,
) -> torch.Tensor:
    """Demand-driven similarity profile computation with MLflow caching.

    1. Look up existing profile run in MLflow by tags.
    2. If found, load from local cache.
    3. If not found, compute profiles, save locally, and log as new MLflow run.

    Returns:
        [N, K] profile tensor where K = num_anchors.
    """
    a_spec_hash = anchors["spec_hash"]
    prof_hash = similarity_profile_hash(run_id, a_spec_hash, similarity_metric, split)

    # Local cache path
    profile_dir = os.path.join(
        artifacts_root, "similarity_profiles", run_id, a_spec_hash, similarity_metric, split
    )
    profile_path = os.path.join(profile_dir, "profiles.pt")

    # 1. Check local cache first
    if os.path.isfile(profile_path):
        return torch.load(profile_path, weights_only=True)

    # 2. Check MLflow for existing run
    existing_run = find_finished_similarity_profile_run(
        experiment_name, run_id, a_spec_hash, similarity_metric, split
    )
    if existing_run is not None:
        # Try to download artifact from MLflow
        try:
            client = mlflow.tracking.MlflowClient()
            local = client.download_artifacts(
                existing_run.info.run_id, "profiles/profiles.pt"
            )
            profiles = torch.load(local, weights_only=True)
            # Cache locally for future use
            os.makedirs(profile_dir, exist_ok=True)
            torch.save(profiles, profile_path)
            return profiles
        except Exception:
            pass  # Fall through to compute

    # 3. Compute from scratch
    backend = get_backend("pt")
    emb_path = os.path.join(artifacts_root, "inference", run_id, split, f"embeddings{backend.extension}")
    if not backend.exists(emb_path):
        raise FileNotFoundError(
            f"HARD FAIL: embeddings not found at {emb_path}. "
            f"Run scripts/02_cache_inference.py first."
        )
    embeddings = backend.load(emb_path)  # [N, D]

    anchor_indices = anchors["anchor_indices"]
    anchor_embeddings = embeddings[anchor_indices]  # [K, D]

    profiles = compute_similarity_profile(embeddings, anchor_embeddings, metric=similarity_metric)

    # Save locally
    os.makedirs(profile_dir, exist_ok=True)
    torch.save(profiles, profile_path)

    # Log as MLflow run (kind=category_similarity_profile)
    tags = category_similarity_profile_tags(
        parent_run_id=run_id,
        anchor_spec_hash=a_spec_hash,
        similarity_metric=similarity_metric,
        split=split,
        profile_hash=prof_hash,
    )
    with mlflow.start_run(
        run_name=f"csp_{similarity_metric}_{run_id[:8]}",
        tags=tags,
    ):
        mlflow.log_params({
            "parent_run_id": run_id,
            "anchor_spec_hash": a_spec_hash,
            "similarity_metric": similarity_metric,
            "split": split,
            "profile_hash": prof_hash,
            "num_anchors": len(anchor_indices),
            "num_samples": int(profiles.shape[0]),
            "profile_dim": int(profiles.shape[1]),
        })
        mlflow.log_artifact(profile_path, artifact_path="profiles")
        log_git_info()

    return profiles


def _assemble_features(
    *,
    run_ids: list,
    artifacts_root: str,
    split: str,
    feature_type: str,
    similarity_metric: str,
    anchors: dict,
    experiment_name: str,
) -> tuple:
    """Assemble features for a meta-learner based on feature_type.

    Feature types:
      - ``logits``:             stacked logits     → [N, M*C]
      - ``embeddings``:         stacked embeddings → [N, M*D]
      - ``embeddings+profiles``: stacked embeddings concatenated with
                                 stacked similarity profiles → [N, M*(D+K)]

    The concatenation order for embeddings+profiles per model is:
        [embedding_i | profile_i]    (embedding first, profile second)
    Then stacked across models:
        [emb_0|prof_0 | emb_1|prof_1 | ... | emb_{M-1}|prof_{M-1}]

    Returns:
        (features_tensor, labels_tensor)
    """
    if feature_type == "logits":
        return _load_features_for_runs(run_ids, artifacts_root, split, "logits")

    elif feature_type == "embeddings":
        return _load_features_for_runs(run_ids, artifacts_root, split, "embeddings")

    elif feature_type == "embeddings+profiles":
        backend = get_backend("pt")
        per_model_features = []
        labels = None

        for run_id in run_ids:
            artifact_dir = os.path.join(artifacts_root, "inference", run_id, split)

            # Load embeddings
            emb_path = os.path.join(artifact_dir, f"embeddings{backend.extension}")
            if not backend.exists(emb_path):
                raise FileNotFoundError(
                    f"HARD FAIL: embeddings not found at {emb_path}. "
                    f"Run scripts/02_cache_inference.py first."
                )
            emb = backend.load(emb_path)  # [N, D]

            # Get or compute similarity profiles (demand-driven)
            profiles = _get_or_compute_similarity_profile(
                run_id=run_id,
                anchors=anchors,
                similarity_metric=similarity_metric,
                split=split,
                artifacts_root=artifacts_root,
                experiment_name=experiment_name,
            )  # [N, K]

            # Concatenate: [embedding | profile] per model
            combined = torch.cat([emb, profiles], dim=1)  # [N, D+K]
            per_model_features.append(combined)

            if labels is None:
                labels_path = os.path.join(artifact_dir, f"labels{backend.extension}")
                if backend.exists(labels_path):
                    labels = backend.load(labels_path)

        if labels is None:
            raise FileNotFoundError("HARD FAIL: could not find labels for any component run.")

        # Stack across models: [N, M*(D+K)]
        stacked = torch.cat(per_model_features, dim=1)
        return stacked, labels

    else:
        raise ValueError(
            f"Unknown feature_type '{feature_type}'. "
            f"Supported: logits, embeddings, embeddings+profiles"
        )


def _three_way_split(N: int, fractions: dict, seed: int):
    """
    Return (train_idx, val_idx, holdout_idx) as numpy arrays.
    fractions: {"train": 0.6, "val": 0.2, "holdout": 0.2}
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    n_train = int(N * fractions["train"])
    n_val = int(N * fractions["val"])
    # remainder goes to holdout
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    holdout_idx = perm[n_train + n_val :]
    return train_idx, val_idx, holdout_idx


def _log_split_membership(manifest, train_idx, val_idx, holdout_idx):
    """Log exact example_id membership for each partition as MLflow artifact."""
    membership = {
        "train": {
            "count": len(train_idx),
            "example_ids": [manifest.example_ids[i] for i in sorted(train_idx)],
        },
        "val": {
            "count": len(val_idx),
            "example_ids": [manifest.example_ids[i] for i in sorted(val_idx)],
        },
        "holdout": {
            "count": len(holdout_idx),
            "example_ids": [manifest.example_ids[i] for i in sorted(holdout_idx)],
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(membership, f, indent=2)
        f.flush()
        mlflow.log_artifact(f.name, artifact_path="meta_split")
        os.unlink(f.name)
    return membership


def _train_adapter(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 256,
):
    """Train adapter on explicit train/val split. Return (best_val_acc, best_state, history)."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_state = None
    n_train = X_train.size(0)

    model.train()
    for ep in range(epochs):
        # Shuffle each epoch
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            xb = X_train[idx]
            yb = y_train[idx]
            out = model(xb)
            loss = loss_fn(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_preds = val_out.argmax(dim=1)
            val_acc = float((val_preds == y_val).float().mean().item())
        model.train()

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    return best_acc, best_state


def _evaluate_holdout(model: nn.Module, X_holdout: torch.Tensor, y_holdout: torch.Tensor) -> float:
    """Evaluate on holdout set. Returns accuracy."""
    model.eval()
    with torch.no_grad():
        out = model(X_holdout)
        preds = out.argmax(dim=1)
        acc = float((preds == y_holdout).float().mean().item())
    return acc


def main():
    """Entry point — redirects to Hydra main."""
    _main()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def _main(cfg: DictConfig) -> None:
    setup_mlflow(cfg)

    split = cfg.pipeline.split
    artifacts_root = cfg.runtime.artifacts_root

    # Seed for reproducibility
    init_seed = cfg.adapter.init_seed
    torch.manual_seed(init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(init_seed)

    # Adapter training hyperparams from Hydra config
    adapter_epochs = cfg.adapter.epochs
    adapter_lr = cfg.adapter.learning_rate
    adapter_batch_size = cfg.adapter.batch_size
    adapter_bias = cfg.adapter.bias
    adapter_dropout = cfg.adapter.dropout

    # Meta-learner identity from Hydra config (sweepable via --multirun)
    meta_type = cfg.adapter.meta_type
    feature_type = cfg.adapter.feature_type
    similarity_metric = cfg.adapter.similarity_metric
    hidden_dim = cfg.adapter.hidden_dim

    # Meta-split config from adapter group
    meta_split_cfg = cfg.adapter.meta_split

    # Dataset manifest for example_id membership logging
    manifest = get_or_create_manifest(
        dataset_name=cfg.dataset.name,
        split=split,
        data_root=cfg.runtime.data_root,
        artifacts_root=artifacts_root,
    )
    manifest_hash = manifest.manifest_hash

    # Device — defaults to cuda via conf/runtime/default.yaml
    dev_name = cfg.runtime.device
    if dev_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev_name)

    # Ensemble definitions from Hydra (no yaml.safe_load)
    ensembles = OmegaConf.to_container(cfg.ensemble.ensembles, resolve=True)

    # Anchor selection (from ensemble defaults → pipeline.anchors)
    anchor_sel = _resolve_anchor_selection(cfg)

    print(f"\nMeta-learner config: type={meta_type} feature_type={feature_type} "
          f"similarity_metric={similarity_metric} bias={adapter_bias}")

    for ens_def in ensembles:
        ens_name = ens_def["name"]
        selector = ens_def.get("selector", {})

        print(f"\n{'='*60}")
        print(f"Meta-learner training for ensemble: {ens_name}")

        # Resolve components — HARD FAIL
        try:
            run_ids = resolve_components(selector, cfg.mlflow.experiment_name)
        except ValueError as e:
            raise RuntimeError(f"HARD FAIL resolving ensemble '{ens_name}': {e}")

        if not run_ids:
            raise RuntimeError(f"HARD FAIL: no component runs for ensemble '{ens_name}'")

        print(f"  Components: {len(run_ids)} runs")

        # ── Anchor setup ──
        anchors = get_or_create_anchors(
            manifest,
            per_class=anchor_sel.get("per_class", 100),
            strategy=anchor_sel.get("strategy", "per_class_first_n"),
            order_by=anchor_sel.get("order_by", "example_id"),
            artifacts_root=artifacts_root,
        )
        a_spec_hash = anchors["spec_hash"]

        # ── Assemble features (demand-driven profiles if needed) ──
        features, labels = _assemble_features(
            run_ids=run_ids,
            artifacts_root=artifacts_root,
            split=split,
            feature_type=feature_type,
            similarity_metric=similarity_metric,
            anchors=anchors,
            experiment_name=cfg.mlflow.experiment_name,
        )
        features = features.to(device)
        labels = labels.to(device)
        in_dim = features.size(1)
        num_classes = int(labels.max().item()) + 1

        # Three-way split
        train_idx, val_idx, holdout_idx = _three_way_split(
            features.size(0),
            {
                "train": float(meta_split_cfg.fractions.train),
                "val": float(meta_split_cfg.fractions.val),
                "holdout": float(meta_split_cfg.fractions.holdout),
            },
            int(meta_split_cfg.seed),
        )

        X_train, y_train = features[train_idx], labels[train_idx]
        X_val, y_val = features[val_idx], labels[val_idx]
        X_holdout, y_holdout = features[holdout_idx], labels[holdout_idx]

        # Build adapter
        if meta_type == "meta_lr":
            adapter = LinearAdapter(
                emb_dim=in_dim, num_classes=num_classes, bias=adapter_bias
            ).to(device)
        elif meta_type == "meta_mlp":
            adapter = ThreeLayerMLPAdapter(
                in_dim=in_dim, num_classes=num_classes,
                hidden_dim=hidden_dim, dropout=adapter_dropout, bias=adapter_bias,
            ).to(device)
        else:
            print(f"  WARN: unknown meta head arch '{meta_type}', skipping.")
            continue

        # Hash computation — includes meta-learner identity
        cs_hash = component_set_hash(run_ids)
        meta_split_spec = json.dumps({
            "seed": int(meta_split_cfg.seed),
            "strategy": str(meta_split_cfg.strategy),
            "fractions": {
                "train": float(meta_split_cfg.fractions.train),
                "val": float(meta_split_cfg.fractions.val),
                "holdout": float(meta_split_cfg.fractions.holdout),
            },
        }, sort_keys=True)
        bi_hash = behavior_input_hash(
            cs_hash,
            split=split,
            feature_type=feature_type,
            anchor_spec=a_spec_hash,
            meta_split_spec=meta_split_spec,
            similarity_metric=similarity_metric,
            init_seed=str(init_seed),
        )

        # Idempotency check
        existing = find_finished_behavior_run(
            cfg.mlflow.experiment_name, bi_hash, behavior=meta_type
        )
        if existing is not None:
            print(f"  {meta_type}: already exists (run_id={existing.info.run_id}). Skipping.")
            continue

        # Train
        best_val_acc, best_state = _train_adapter(
            adapter, X_train, y_train, X_val, y_val,
            epochs=adapter_epochs,
            lr=adapter_lr,
            batch_size=adapter_batch_size,
        )

        # Evaluate on holdout
        if best_state is not None:
            adapter.load_state_dict(best_state)
        holdout_acc = _evaluate_holdout(adapter, X_holdout, y_holdout)

        # Tags
        name_parts = [f"adapter_{ens_name}_{meta_type}"]
        if meta_type == "meta_mlp":
            name_parts.append(f"h{hidden_dim}")

        tags = behavior_tags(
            behavior=meta_type,
            component_run_ids=run_ids,
            behavior_input_hash=bi_hash,
            component_set_hash=cs_hash,
            extra={
                "ensemble_name": ens_name,
                "split": split,
                "feature_type": feature_type,
                "similarity_metric": similarity_metric,
                "kind": "behavior",
                "meta_type": meta_type,
                "dataset_manifest_hash": manifest_hash,
                "anchor_spec_hash": a_spec_hash,
                "adapter_bias": str(adapter_bias),
                "adapter_arch": meta_type,
                "init_seed": str(init_seed),
            },
        )

        with mlflow.start_run(run_name="_".join(name_parts), tags=tags) as run:
            mlflow.log_params({
                "ensemble_name": ens_name,
                "adapter_type": meta_type,
                "method_type": "meta",
                "num_components": len(run_ids),
                "in_dim": in_dim,
                "feature_type": feature_type,
                "similarity_metric": similarity_metric,
                "split": split,
                "epochs": adapter_epochs,
                "learning_rate": adapter_lr,
                "batch_size": adapter_batch_size,
                "adapter_bias": adapter_bias,
                "adapter_dropout": adapter_dropout,
                "anchor_spec_hash": a_spec_hash,
                "meta_split_seed": int(meta_split_cfg.seed),
                "meta_split_strategy": str(meta_split_cfg.strategy),
                "meta_split_train_frac": float(meta_split_cfg.fractions.train),
                "meta_split_val_frac": float(meta_split_cfg.fractions.val),
                "meta_split_holdout_frac": float(meta_split_cfg.fractions.holdout),
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_holdout": len(holdout_idx),
                "dataset_manifest_hash": manifest_hash,
                "init_seed": init_seed,
            })
            if meta_type == "meta_mlp":
                mlflow.log_params({"hidden_dim": hidden_dim})

            mlflow.log_metric("adapter_val_accuracy", best_val_acc)
            mlflow.log_metric("adapter_holdout_accuracy", holdout_acc)

            # Log example_id membership
            _log_split_membership(manifest, train_idx, val_idx, holdout_idx)

            # Log adapter weights
            if best_state is not None:
                with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                    torch.save(best_state, f.name)
                    mlflow.log_artifact(f.name, artifact_path="adapter")
                    os.unlink(f.name)

            # Log component IDs
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump({"component_run_ids": run_ids}, f, indent=2)
                f.flush()
                mlflow.log_artifact(f.name, artifact_path="adapter")
                os.unlink(f.name)

            log_git_info()
            log_resolved_config(cfg)

            print(f"  {meta_type}: val_acc={best_val_acc:.4f} holdout_acc={holdout_acc:.4f}  run_id={run.info.run_id}")

    print("\nMeta-learner training complete.")


if __name__ == "__main__":
    main()
