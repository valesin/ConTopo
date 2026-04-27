#!/usr/bin/env python3
"""
03b_compute_diagnostics.py — Optional per-model diagnostic metrics.

For the FINISHED model run matching the current configuration, compute spatial
and weight-based diagnostics:
  - Moran's I (spatial autocorrelation of embeddings on the topographic grid)
  - Weight norms (L2 norm of each output unit's weight vector)
  - Unit distance correlation (grid distance vs weight cosine similarity)

Each diagnostic is gated by ``pipeline.diagnostics.*`` config flags.

**One MLflow run per (model, metric)**: each diagnostic produces its own
MLflow run (kind=diagnostics).

Usage:
    python scripts/03b_compute_diagnostics.py
    python scripts/03b_compute_diagnostics.py profiling.diagnostics.morans_i=false
    python scripts/03b_compute_diagnostics.py execution.force=true
"""

from __future__ import annotations

import os
import hydra
import mlflow
import torch
import torch.nn as nn
from omegaconf import DictConfig

import tempfile
from src.config.hash import identity_hash
from src.mlflow_utils import (
    apply_mlflow_env_overrides,
    setup_mlflow,
    resolve_seed,
    resolve_device,
    load_mlflow_artifact,
    get_run_context,
    log_dataset_lineage,
)
from src.repositories.functional_run_repository import (
    configure_run_repository,
    find_finished_model_run,
    find_finished_identity_run,
)
from src.profiling.smoothness import morans_i
from src.profiling.unit_analysis import weight_norms, unit_distance_correlation
from src.networks.registry import unwrap
from src.data.loaders import get_split_labels
from src.mlflow_schema_logger import (
    log_params as schema_log_params,
    start_run as schema_start_run,
    timed_log_metric,
    timed_log_artifact,
)


def _log_diagnostic_run(
    run_id,
    model_params,
    model_tags,
    inf_run_id,
    metric_name,
    metrics,
    artifact_dir,
    cfg,
    split,
):
    """Create one MLflow run for a single diagnostic metric."""
    parent_run = mlflow.get_run(run_id)
    rho, trial, topology = get_run_context(parent_run)

    tags = {
        "parent_run_id": run_id,
        "identity_hash": identity_hash(
            "diagnostics",
            parent_run_id=run_id,
            diagnostic_metric=metric_name,
            split=split,
        ),
        "run_name": f"diag_{metric_name}_{topology}_rho{rho}_t{trial}",
    }
    if inf_run_id:
        tags["inference_run_id"] = inf_run_id

    with schema_start_run(
        kind="diagnostics",
        run_name=f"diag_{metric_name}_{topology}_rho{rho}_t{trial}",
        tags=tags,
    ) as diag_run:
        params = {
            "diagnostic_metric": metric_name,
            "split": cfg.execution.split,
        }

        schema_log_params("diagnostics", params)

        for k, v in metrics.items():
            timed_log_metric(k, v)

        log_dataset_lineage(
            get_split_labels(cfg, cfg.execution.split),
            cfg.execution.split,
            cfg.dataset.name,
            context="diagnostics",
        )

        # Log applicable artifact files for this metric
        if artifact_dir and os.path.isdir(artifact_dir):
            for fname in os.listdir(artifact_dir):
                if fname.startswith(metric_name):
                    fpath = os.path.join(artifact_dir, fname)
                    if os.path.isfile(fpath):
                        timed_log_artifact(fpath, artifact_path="diagnostics")

    metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    print(f"    {metric_name}: {metric_str}  run_id={diag_run.info.run_id}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    apply_mlflow_env_overrides(cfg)
    setup_mlflow(cfg)
    configure_run_repository(cfg.mlflow.tracking_uri, cfg.mlflow.experiment_name)

    # ── Seed ──
    seed = resolve_seed(cfg)
    cfg.seed = seed

    diag_cfg = cfg.profiling.diagnostics
    enabled = []
    if diag_cfg.morans_i:
        enabled.append("morans_i")
    if diag_cfg.weight_norms:
        enabled.append("weight_norms")
    if diag_cfg.unit_distance_correlation:
        enabled.append("unit_distance_correlation")

    if not enabled:
        print("No diagnostics enabled. Nothing to do.")
        return

    # ── Target specific model by ID ──
    model_run, _ = find_finished_model_run(cfg, seed)

    if model_run is None:
        print(
            "No trained model found for this config. Please run 01_train_models.py first."
        )
        return

    run_id = model_run.info.run_id

    split = cfg.execution.split
    force = cfg.execution.force
    device = resolve_device(cfg.runtime.device)

    # Get parent model params for logging
    model_run = mlflow.get_run(run_id)
    model_params = model_run.data.params
    model_tags = model_run.data.tags
    rho, trial, _ = get_run_context(model_run)

    print(f"Found target model Run ID: {run_id}, rho={rho} trial={trial}")
    print(f"Enabled diagnostics: {enabled}")

    # ── Check what needs to be computed ──
    needed = []
    for metric_name in enabled:
        if not force:
            existing = find_finished_identity_run(
                "diagnostics",
                identity_hash(
                    "diagnostics",
                    parent_run_id=run_id,
                    diagnostic_metric=metric_name,
                    split=split,
                ),
            )
            if existing is not None:
                print(f"  -> Diagnostic {metric_name} already computed. Skipping.")
                continue
        needed.append(metric_name)

    if not needed:
        print("All requested diagnostics are already computed.")
        return

    print(f"Computing: {needed}")

    # ── Fetch corresponding Inference Run (for Moran's I) ──
    inf_run_id = None
    if "morans_i" in needed:
        inf_identity = identity_hash(
            "inference", trained_model_run_id=run_id, split=split
        )
        inf_run = find_finished_identity_run("inference", inf_identity)

        if inf_run is not None:
            inf_run_id = inf_run.info.run_id
        else:
            print(
                f"  WARN: No inference run found for split '{split}'. Skipping morans_i."
            )
            needed.remove("morans_i")

    with tempfile.TemporaryDirectory() as diag_dir:
        # ── Moran's I ──
        if "morans_i" in needed:
            data = load_mlflow_artifact(
                inf_run_id,
                f"inference/{split}_tensors.npz",
                file_type="numpy",
                strict=True,
                cache_dir=cfg.mlflow.artifact_cache_dir,
            )
            embs = torch.from_numpy(data["embeddings"])

            emb_dim = int(model_run.data.params.get("embedding_dim", 256))
            mi = morans_i(embs, emb_dim)

            _log_diagnostic_run(
                run_id,
                model_params,
                model_tags,
                inf_run_id,
                "morans_i",
                {"morans_i": mi},
                diag_dir,
                cfg,
                split,
            )

        # ── Weight-based diagnostics ──
        weight_needed = [
            m for m in needed if m in ("weight_norms", "unit_distance_correlation")
        ]
        if weight_needed:
            # Load model weights only when needed for weight-based diagnostics
            model_uri = f"runs:/{run_id}/e2e_best"
            try:
                model = mlflow.pytorch.load_model(model_uri)
            except Exception as e:
                print(f"ERROR: failed to load model {model_uri}: {e}")
                raise
            # Safely unwrap model to target the embedding linear layer.
            # FinetuneResNet34/ScratchResNet34 use `neck`; LinearResNet18 uses `fc`.
            base_model = unwrap(model).to(device)
            fc_layer = getattr(base_model, "neck", None) or getattr(
                base_model, "fc", None
            )

            if fc_layer is None or not isinstance(fc_layer, nn.Linear):
                print(
                    "    WARN: fc_layer is not nn.Linear (or not found), skipping weight diagnostics"
                )
            else:
                if "weight_norms" in weight_needed:
                    wnorms = weight_norms(fc_layer)
                    torch.save(wnorms, os.path.join(diag_dir, "weight_norms.pt"))
                    _log_diagnostic_run(
                        run_id,
                        model_params,
                        model_tags,
                        None,
                        "weight_norms",
                        {
                            "weight_norms_mean": float(wnorms.mean()),
                            "weight_norms_std": float(wnorms.std()),
                        },
                        diag_dir,
                        cfg,
                        split,
                    )

                if "unit_distance_correlation" in weight_needed:
                    udc = unit_distance_correlation(fc_layer)
                    torch.save(
                        udc, os.path.join(diag_dir, "unit_distance_correlation.pt")
                    )
                    metrics = {}
                    if udc.shape[0] > 2:
                        from src.profiling.rdm import pearson_corrcoef

                        r = float(pearson_corrcoef(udc.t())[0, 1].item())
                        metrics["unit_dist_cos_correlation"] = r
                    _log_diagnostic_run(
                        run_id,
                        model_params,
                        model_tags,
                        None,
                        "unit_distance_correlation",
                        metrics,
                        diag_dir,
                        cfg,
                        split,
                    )

    print("\nDone.")


if __name__ == "__main__":
    main()
