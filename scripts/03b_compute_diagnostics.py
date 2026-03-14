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
    python scripts/03b_compute_diagnostics.py pipeline.diagnostics.morans_i=false
    python scripts/03b_compute_diagnostics.py pipeline.force=true
"""

from __future__ import annotations

import os
import hydra
import mlflow
import mlflow.artifacts
import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.config.paths import get_cache_dir
from src.config.hash import cfg_hash
from src.data.manifest import get_or_create_manifest
from src.mlflow_utils import (
    log_resolved_config,
    setup_mlflow,
    get_existing_model,
    resolve_seed,
    get_inference_run,
    load_mlflow_artifact,
    log_manifest_lineage,
    find_finished_diagnostic_run,
)
from src.profiling.smoothness import morans_i
from src.profiling.unit_analysis import weight_norms, unit_distance_correlation
from src.networks.registry import unwrap





def _log_diagnostic_run(
    run_id,
    model_tags,
    inf_run_id,
    metric_name,
    metrics,
    artifact_dir,
    cfg,
    manifest=None,
):
    """Create one MLflow run for a single diagnostic metric."""
    rho = model_tags.get("rho", "?")
    trial = model_tags.get("trial", "?")
    topology = model_tags.get("topology", "?")

    tags = {
        "kind": "diagnostics",
        "parent_run_id": run_id,
        "diagnostic_metric": metric_name,
        "rho": rho,
        "trial": trial,
        "topology": topology,
    }
    if inf_run_id:
        tags["inference_run_id"] = inf_run_id

    with mlflow.start_run(
        run_name=f"diag_{metric_name}_{topology}_rho{rho}_t{trial}",
        tags=tags,
    ) as diag_run:
        params = {
            "parent_run_id": run_id,
            "diagnostic_metric": metric_name,
            "split": cfg.pipeline.split,
        }
        if inf_run_id:
            params["inference_run_id"] = inf_run_id

        mlflow.log_params(params)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # ── Log the dataset manifest to MLflow for lineage ──
        if manifest is not None:
            log_manifest_lineage(manifest, cfg.pipeline.split, cfg.dataset.name, context="diagnostics")

        # Log applicable artifact files for this metric
        if artifact_dir and os.path.isdir(artifact_dir):
            for fname in os.listdir(artifact_dir):
                if fname.startswith(metric_name):
                    fpath = os.path.join(artifact_dir, fname)
                    if os.path.isfile(fpath):
                        mlflow.log_artifact(fpath, artifact_path="diagnostics")

        log_resolved_config(cfg)

    metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    print(f"    {metric_name}: {metric_str}  run_id={diag_run.info.run_id}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_mlflow(cfg)

    # ── Seed ──
    seed = resolve_seed(cfg)
    cfg.seed = seed

    diag_cfg = cfg.pipeline.diagnostics
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
    hash_val = cfg_hash(cfg)
    model, run_id = get_existing_model(cfg.mlflow.experiment_name, hash_val)

    if run_id is None:
        print(
            f"A model with cfg_hash={hash_val} has not been trained yet. Please run 01_train_models.py first."
        )
        return

    split = cfg.pipeline.split
    force = cfg.pipeline.force
    cache_dir = get_cache_dir(cfg)
    artifacts_root = str(cache_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Manifest ──
    manifest = get_or_create_manifest(
        dataset_name=cfg.dataset.name,
        split=split,
        data_root=cfg.runtime.data_root,
        artifacts_root=artifacts_root,
    )

    # Get parent model tags for logging
    model_run = mlflow.get_run(run_id)
    model_tags = model_run.data.tags
    rho = model_tags.get("rho", "?")
    trial = model_tags.get("trial", "?")

    print(f"Found target model Run ID: {run_id}, rho={rho} trial={trial}")
    print(f"Enabled diagnostics: {enabled}")

    # ── Check what needs to be computed ──
    needed = []
    for metric_name in enabled:
        if not force:
            existing = find_finished_diagnostic_run(
                cfg.mlflow.experiment_name, run_id, metric_name
            )
            if existing is not None:
                print(f"  -> Diagnostic {metric_name} already computed. Skipping.")
                continue
        needed.append(metric_name)

    if not needed:
        print("All requested diagnostics are already computed.")
        return

    print(f"Computing: {needed}")

    diag_dir = os.path.join(artifacts_root, "diagnostics", run_id)
    os.makedirs(diag_dir, exist_ok=True)

    # ── Fetch corresponding Inference Run (for Moran's I) ──
    inf_run_id = None
    if "morans_i" in needed:
        inf_runs = get_inference_run(cfg.mlflow.experiment_name, run_id, split)

        if len(inf_runs) > 0:
            inf_run_id = inf_runs.iloc[0].run_id
        else:
            print(
                f"  WARN: No inference run found for split '{split}'. Skipping morans_i."
            )
            needed.remove("morans_i")

    # ── Moran's I ──
    if "morans_i" in needed:
        data = load_mlflow_artifact(inf_run_id, f"inference_data/{split}_tensors.npz", file_type="numpy", strict=True)
        embs = torch.from_numpy(data["embeddings"])

        emb_dim = int(model_run.data.params.get("embedding_dim", 256))
        mi = morans_i(embs, emb_dim)

        _log_diagnostic_run(
            run_id,
            model_tags,
            inf_run_id,
            "morans_i",
            {"morans_i": mi},
            diag_dir,
            cfg,
        )

    # ── Weight-based diagnostics ──
    weight_needed = [
        m for m in needed if m in ("weight_norms", "unit_distance_correlation")
    ]
    if weight_needed:
        # Safely unwrap model to target the `fc` layer directly
        base_model = unwrap(model).to(device)
        fc_layer = getattr(base_model, "fc", None)

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
                    model_tags,
                    None,
                    "weight_norms",
                    {
                        "weight_norms_mean": float(wnorms.mean()),
                        "weight_norms_std": float(wnorms.std()),
                    },
                    diag_dir,
                    cfg,
                )

            if "unit_distance_correlation" in weight_needed:
                udc = unit_distance_correlation(fc_layer)
                torch.save(udc, os.path.join(diag_dir, "unit_distance_correlation.pt"))
                metrics = {}
                if udc.shape[0] > 2:
                    from src.profiling.rdm import pearson_corrcoef

                    r = float(pearson_corrcoef(udc.t())[0, 1].item())
                    metrics["unit_dist_cos_correlation"] = r
                _log_diagnostic_run(
                    run_id,
                    model_tags,
                    None,
                    "unit_distance_correlation",
                    metrics,
                    diag_dir,
                    cfg,
                )

    print("\nDone.")


if __name__ == "__main__":
    main()
