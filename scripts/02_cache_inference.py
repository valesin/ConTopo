#!/usr/bin/env python3
"""
02_cache_inference.py — Hydra + MLflow inference caching.

For each FINISHED model run in MLflow, run inference on the eval split,
save artifacts (logits, preds, probs, embeddings, labels, hashes,
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
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.data.loaders import get_cifar10_eval_loader
from src.data.manifest import get_or_create_manifest
from src.config.paths import get_cache_dir
from src.inference import run_combined_model_inference
from src.mlflow_utils import (
    log_resolved_config,
    setup_mlflow,
    get_existing_model,
    resolve_seed,
    log_manifest_lineage,
    find_finished_run,
    safe_to_numpy_float64,
    get_inference_run,
)
from src.config.hash import cfg_hash


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_mlflow(cfg)

    # ── Seed ──
    seed = resolve_seed(cfg)
    cfg.seed = seed

    # ── Idempotency ──
    hash_val = cfg_hash(cfg)

    model, run_id = get_existing_model(cfg.mlflow.experiment_name, hash_val)
    if model is None:
        print(
            f"A model with cfg_hash={hash_val} has not been trained yet. Please run 01_train_models.py first with the same config (or set pipeline.force=true to ignore this check)."
        )
        return

    # Extract parent run metadata
    parent_run = mlflow.get_run(run_id)
    parent_run_name = parent_run.info.run_name

    # Try getting 'rho' from parent run's parameters
    rho = parent_run.data.params.get("rho", "N/A")

    split = cfg.pipeline.split
    cache_dir = get_cache_dir(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = get_or_create_manifest(
        dataset_name=cfg.dataset.name,
        split=split,
        data_root=cfg.runtime.data_root,
        artifacts_root=str(cache_dir),
    )
    manifest_hash = manifest.manifest_hash

    # Check if this specific inference run already exists
    if not cfg.pipeline.force:
        inf_runs = get_inference_run(cfg.mlflow.experiment_name, run_id, split)
        if not inf_runs.empty:
            existing_hash = inf_runs.iloc[0].get("tags.dataset_manifest_hash", "")
            if existing_hash == manifest_hash:
                print(f"Inference already cached for model {run_id} on split {split}. Skipping.")
                return

    loader = get_cifar10_eval_loader(
        root=cfg.runtime.data_root,
        batch_size=cfg.runtime.inference.batch_size,
        num_workers=cfg.runtime.inference.num_workers,
        preset=cfg.dataset.transforms.preset,
    )

    # Run inference
    results = run_combined_model_inference(model, loader, device)

    # Merge the manifest tracking data
    results["hashes"] = manifest.hashes
    results["original_indices"] = manifest.original_indices
    results["labels"] = manifest.labels

    tags = {
        "kind": "inference",
        "split": split,
        "dataset_manifest_hash": manifest_hash,
        "trained_model_run_id": run_id,  # Link back to the trained model
        "parent_run_name": parent_run_name,
    }

    with mlflow.start_run(
        run_name=f"inf_{split}_{parent_run_name}", tags=tags
    ) as inf_run:
        mlflow.log_params(
            {
                "dataset": cfg.dataset.name,
                "split": split,
                "dataset_manifest_hash": manifest_hash,
                "transforms_preset": cfg.dataset.transforms.preset,
                "trained_model_run_id": run_id,
                "parent_run_name": parent_run_name,
                "rho": rho,
                "cfg_hash": hash_val,
            }
        )

        # ─── 1. Save Tabular Data (IDs, Labels, Preds) ───

        # Convert tensors to numpy upfront to avoid pandas broadcast errors natively and cast int to float64 to avoid MLflow schema warning
        eval_df = pd.DataFrame(
            {
                "example_id": results["hashes"],
                "original_index": safe_to_numpy_float64(results["original_indices"]),
                "label": safe_to_numpy_float64(results["labels"]),
                "prediction": safe_to_numpy_float64(results["preds"]),
                "confidence": safe_to_numpy_float64(
                    results["probs"].numpy().max(axis=1) if hasattr(results["probs"], "numpy") else results["probs"].max(axis=1)
                ),
            }
        )

        # Log ONLY the inputs/targets to MLflow Dataset for lineage
        log_manifest_lineage(manifest, split, cfg.dataset.name, context="evaluation")

        # Save and log the Parquet file (this contains the predictions -> artifacts)
        os.makedirs(cache_dir, exist_ok=True)
        tabular_path = os.path.join(cache_dir, f"{split}_inference_results.parquet")
        eval_df.to_parquet(tabular_path, index=False)
        mlflow.log_artifact(tabular_path, artifact_path="inference_data")

        # ─── 2. Save Heavy Matrices (Embeddings, Logits, Probs) ───
        tensors_path = os.path.join(cache_dir, f"{split}_tensors.npz")
        np.savez_compressed(
            tensors_path,
            embeddings=results["embeddings"],
            logits=results["logits"],
            probs=results["probs"],  # The full Nx10 probability matrix
        )
        mlflow.log_artifact(tensors_path, artifact_path="inference_data")

        # ─── 3. Quick Accuracy Logging ───
        # Since we skipped evaluate(), let's manually log just the accuracy
        preds_np = (
            results["preds"].numpy()
            if hasattr(results["preds"], "numpy")
            else results["preds"]
        )
        labels_np = (
            results["labels"].numpy()
            if hasattr(results["labels"], "numpy")
            else results["labels"]
        )
        acc = float((preds_np == labels_np).mean())
        mlflow.log_metric("accuracy", acc)

        print(f"Inference cached! Accuracy: {acc:.4f}")

        log_resolved_config(cfg)

    print("Done.")


if __name__ == "__main__":
    main()
