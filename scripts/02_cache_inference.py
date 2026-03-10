#!/usr/bin/env python3
"""
02_cache_inference.py — Hydra + MLflow inference caching.

For each FINISHED model run in MLflow, run inference on the eval split,
save artifacts (logits, preds, probs, embeddings, labels, example_ids,
original_indices) locally and as MLflow artifacts.  Each inference
operation is tracked as its own MLflow run (kind=inference).

Usage:
    python scripts/02_cache_inference.py
    python scripts/02_cache_inference.py pipeline.split=val
    python scripts/02_cache_inference.py pipeline.force=true
"""

from __future__ import annotations

import os

import hydra
import mlflow
import torch
from omegaconf import DictConfig

from src.data.loaders import get_cifar10_eval_loader
from src.data.manifest import get_or_create_manifest
from src.inference import (
    ARTIFACT_KEYS,
    artifacts_complete,
    run_combined_model_inference,
    save_inference_artifacts,
)
from src.data.cache import get_backend
from src.mlflow_utils import (
    find_finished_inference_run,
    log_git_info,
    log_resolved_config,
    setup_mlflow,
)
from src.networks.resnet18 import LinearResNet18
from src.training.checkpoint import load_checkpoint


def _load_model_from_run(run, device):
    """Download checkpoint artifact and reconstruct model."""
    client = mlflow.tracking.MlflowClient()
    run_id = run.info.run_id

    artifacts = client.list_artifacts(run_id, path="checkpoint")
    ckpt_artifact = None
    for a in artifacts:
        if a.path.endswith(".pth"):
            ckpt_artifact = a
            break

    if ckpt_artifact is None:
        return None

    local_path = client.download_artifacts(run_id, ckpt_artifact.path)
    ckpt = load_checkpoint(local_path, device=device)

    params = run.data.params
    emb_dim = int(params.get("embedding_dim", 256))
    num_classes = int(params.get("num_classes", 10))
    p_dropout = float(params.get("p_dropout", 0.5))
    head_bias = params.get("head_bias", "True").lower() in ("true", "1", "yes")

    model = LinearResNet18(
        emb_dim=emb_dim,
        num_classes=num_classes,
        p_dropout=p_dropout,
        use_dropout=True,
        ret_emb=True,
        head_bias=head_bias,
    )

    state_dict = ckpt.get("state_dict", {})
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_mlflow(cfg)

    split = cfg.pipeline.split
    force = cfg.pipeline.force
    backend = get_backend(cfg.runtime.storage.backend)
    artifacts_root = cfg.runtime.artifacts_root

    exp = mlflow.get_experiment_by_name(cfg.mlflow.experiment_name)
    if exp is None:
        print(f"Experiment '{cfg.mlflow.experiment_name}' not found.")
        return

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.kind = 'model' and attributes.status = 'FINISHED'",
        output_format="list",
    )
    print(f"Found {len(runs)} model runs.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset manifest
    manifest = get_or_create_manifest(
        dataset_name=cfg.dataset.name,
        split=split,
        data_root=cfg.runtime.data_root,
        artifacts_root=artifacts_root,
    )
    manifest_hash = manifest.manifest_hash

    loader = get_cifar10_eval_loader(
        root=cfg.runtime.data_root,
        batch_size=cfg.runtime.inference.batch_size,
        num_workers=cfg.runtime.inference.num_workers,
        preset=cfg.dataset.transforms.preset,
    )

    cached = computed = failed = 0

    for run in runs:
        run_id = run.info.run_id
        rho = run.data.tags.get("rho", "?")
        trial = run.data.tags.get("trial", "?")
        topology = run.data.tags.get("topology", "?")
        artifact_dir = os.path.join(artifacts_root, "inference", run_id, split)

        # Idempotency: check if inference run already exists
        if not force:
            existing = find_finished_inference_run(
                cfg.mlflow.experiment_name, run_id, split
            )
            if existing is not None:
                cached += 1
                continue
            # Also check local cache
            if artifacts_complete(artifact_dir, backend):
                cached += 1
                continue

        print(f"Processing topology={topology} rho={rho} trial={trial} run_id={run_id}...")
        model = _load_model_from_run(run, device)
        if model is None:
            print("  SKIP: no checkpoint artifact found.")
            failed += 1
            continue

        results = run_combined_model_inference(model, loader, device)
        results["example_ids"] = manifest.example_ids
        results["original_indices"] = manifest.original_indices
        results["labels"] = manifest.labels

        # Save locally
        save_inference_artifacts(results, artifact_dir, backend)

        # Create a tracked MLflow run for this inference step
        tags = {
            "kind": "inference",
            "parent_run_id": run_id,
            "parent_cfg_hash": run.data.tags.get("cfg_hash", ""),
            "split": split,
            "dataset_manifest_hash": manifest_hash,
            "rho": rho,
            "trial": trial,
            "topology": topology,
        }

        with mlflow.start_run(
            run_name=f"infer_{topology}_rho{rho}/trial_{trial}",
            tags=tags,
        ) as inf_run:
            mlflow.log_params({
                "parent_run_id": run_id,
                "split": split,
                "dataset_manifest_hash": manifest_hash,
                "backend": cfg.runtime.storage.backend,
                "transforms_preset": cfg.dataset.transforms.preset,
            })
            mlflow.log_metric(f"{split}_accuracy", results["accuracy"])

            for key in ARTIFACT_KEYS:
                local_file = os.path.join(artifact_dir, f"{key}{backend.extension}")
                if os.path.isfile(local_file):
                    mlflow.log_artifact(local_file, artifact_path=f"inference/{split}")

            log_git_info()
            log_resolved_config(cfg)

        computed += 1
        print(f"  Done. accuracy={results['accuracy']:.4f}  run_id={inf_run.info.run_id}")

    print(f"\nTotal: {len(runs)}  Cached: {cached}  Computed: {computed}  Failed: {failed}")


if __name__ == "__main__":
    main()
