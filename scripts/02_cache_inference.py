#!/usr/bin/env python3
"""
02_cache_inference.py — Hydra + MLflow inference caching.

For each FINISHED model run in MLflow, run inference on the eval split,
save artifacts (logits, preds, probs, embeddings, labels) locally and
as MLflow artifacts.  Each inference operation is tracked as its own
MLflow run (kind=inference).

Usage:
    python scripts/02_cache_inference.py
    python scripts/02_cache_inference.py execution.split=val
    python scripts/02_cache_inference.py execution.force=true
"""

from __future__ import annotations
import os
import tempfile
import hydra
import mlflow
import torch
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.data.loaders import get_cifar10_eval_loader, shutdown_dataloader_workers
from src.inference import run_combined_model_inference
from src.mlflow_utils import (
    log_resolved_config,
    setup_mlflow,
    find_finished_model_run,
    resolve_seed,
    resolve_device,
    get_run_context,
    safe_to_numpy_float64,
    get_inference_run,
    log_dataset_lineage,
)
from src.config.hash import cfg_hash, identity_hash
from src.mlflow_schema_logger import (
    log_params as schema_log_params,
    start_run as schema_start_run,
)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_mlflow(cfg)

    # ── Seed ──
    seed = resolve_seed(cfg)
    cfg.seed = seed

    # ── Find parent model run ──
    hash_val = cfg_hash(cfg)  # kept for tagging the inference run
    model_run, model_hash = find_finished_model_run(cfg.mlflow.experiment_name, cfg, seed)
    if model_run is None:
        print(
            f"No trained model found (identity_hash={model_hash}). "
            "Please run 01_train_models.py first with the same config."
        )
        return
    run_id = model_run.info.run_id
    model_uri = f"runs:/{run_id}/e2e_best"
    print(f"Loading model weights from {model_uri}...")
    model = mlflow.pytorch.load_model(model_uri)

    # Extract parent run metadata
    parent_run = mlflow.get_run(run_id)
    parent_run_name = parent_run.info.run_name
    rho, _, _ = get_run_context(parent_run)
    if rho == "?":
        rho = "N/A"

    split = cfg.execution.split
    device = resolve_device(cfg.runtime.device)

    # Check if this specific inference run already exists
    if not cfg.execution.force:
        inf_runs = get_inference_run(cfg.mlflow.experiment_name, run_id, split)
        if not inf_runs.empty:
            print(
                f"Inference already cached for model {run_id} on split {split}. Skipping."
            )
            return

    loader = get_cifar10_eval_loader(
        root=cfg.runtime.data_root,
        batch_size=cfg.runtime.inference.batch_size,
        num_workers=cfg.runtime.inference.num_workers,
        preset=cfg.dataset.transforms.preset,
        split=split,
        val_per_class=cfg.dataset.split.val_per_class,
    )

    try:
        # Run inference
        results = run_combined_model_inference(model, loader, device)

        tags = {
            "trained_model_run_id": run_id,  # Link back to the trained model
            "parent_run_name": parent_run_name,
            "cfg_hash": hash_val,
            "identity_hash": identity_hash(
                "inference", trained_model_run_id=run_id, split=split
            ),
        }

        with schema_start_run(
            kind="inference", run_name=f"inf_{split}_{parent_run_name}", tags=tags
        ) as inf_run:
            schema_log_params(
                "inference",
                {
                    "dataset": cfg.dataset.name,
                    "split": split,
                    "transforms_preset": cfg.dataset.transforms.preset,
                    "rho": rho,
                },
            )

            log_dataset_lineage(
                results["labels"], split, cfg.dataset.name, context="evaluation"
            )

            # ─── 1. Save Tabular Data (Labels, Preds) ───
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

            eval_df = pd.DataFrame(
                {
                    "original_index": safe_to_numpy_float64(
                        torch.arange(len(labels_np))
                    ),
                    "label": safe_to_numpy_float64(results["labels"]),
                    "prediction": safe_to_numpy_float64(results["preds"]),
                    "confidence": safe_to_numpy_float64(
                        results["probs"].numpy().max(axis=1)
                        if hasattr(results["probs"], "numpy")
                        else results["probs"].max(axis=1)
                    ),
                }
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                # Save and log the Parquet file (predictions -> artifacts)
                tabular_path = os.path.join(
                    tmpdir, f"{split}_inference_results.parquet"
                )
                eval_df.to_parquet(tabular_path, index=False)
                mlflow.log_artifact(tabular_path, artifact_path="inference_data")

                # ─── 2. Save Heavy Matrices (Embeddings, Logits, Probs) ───
                tensors_path = os.path.join(tmpdir, f"{split}_tensors.npz")
                np.savez_compressed(
                    tensors_path,
                    embeddings=results["embeddings"],
                    logits=results["logits"],
                    probs=results["probs"],  # The full Nx10 probability matrix
                )
                mlflow.log_artifact(tensors_path, artifact_path="inference_data")

            # ─── 3. Quick Accuracy Logging ───
            acc = float((preds_np == labels_np).mean())
            mlflow.log_metric("accuracy", acc)

            print(f"Inference cached! Accuracy: {acc:.4f}")

            log_resolved_config(cfg)
    finally:
        shutdown_dataloader_workers(loader)

    print("Done.")


if __name__ == "__main__":
    main()
