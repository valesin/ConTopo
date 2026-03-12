#!/usr/bin/env python3
"""
03b_compute_diagnostics.py — Optional per-model diagnostic metrics.

For each FINISHED model run, compute spatial and weight-based diagnostics:
  - Moran's I (spatial autocorrelation of embeddings on the topographic grid)
  - Weight norms (L2 norm of each output unit's weight vector)
  - Unit distance correlation (grid distance vs weight cosine similarity)

Each diagnostic is gated by ``pipeline.diagnostics.*`` config flags.

**One MLflow run per (model, metric)**: each diagnostic produces its own
MLflow run (kind=diagnostics).  Adding a new metric later only computes
the missing one — previously computed metrics are never recalculated.

Usage:
    python scripts/03b_compute_diagnostics.py
    python scripts/03b_compute_diagnostics.py pipeline.diagnostics.morans_i=false
    python scripts/03b_compute_diagnostics.py pipeline.force=true
"""

from __future__ import annotations

import os

import hydra
import mlflow
import torch
from omegaconf import DictConfig

from src.data.cache import get_backend
from src.config.paths import get_cache_dir
from src.mlflow_utils import (
    log_git_info,
    log_resolved_config,
    setup_mlflow,
)
from src.profiling.smoothness import morans_i
from src.profiling.unit_analysis import weight_norms, unit_distance_correlation
from src.training.checkpoint import load_checkpoint
from src.networks.resnet18 import LinearResNet18


def _find_finished_diagnostic_run(experiment_name: str, parent_run_id: str, metric_name: str):
    """Check if a diagnostic run already exists for this (model, metric)."""
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return None
    filter_str = (
        f"tags.kind = 'diagnostics' and "
        f"tags.parent_run_id = '{parent_run_id}' and "
        f"tags.diagnostic_metric = '{metric_name}' and "
        f"attributes.status = 'FINISHED'"
    )
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=filter_str,
        max_results=1,
        output_format="list",
    )
    return runs[0] if runs else None


def _load_encoder_and_fc(run, device):
    """Reconstruct encoder + FC layer from MLflow run checkpoint."""
    client = mlflow.tracking.MlflowClient()
    run_id = run.info.run_id
    artifacts = client.list_artifacts(run_id, path="checkpoint")
    ckpt_path = None
    for a in artifacts:
        if a.path.endswith(".pth"):
            ckpt_path = client.download_artifacts(run_id, a.path)
            break
    if ckpt_path is None:
        return None, None

    ckpt = load_checkpoint(ckpt_path, device=device)
    params = run.data.params
    emb_dim = int(params.get("embedding_dim", 256))

    model = LinearResNet18(emb_dim=emb_dim, num_classes=10, ret_emb=True)
    state_dict = ckpt.get("state_dict", {})
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model.encoder, model.fc


def _log_diagnostic_run(
    run, metric_name, metrics, artifact_dir, cfg,
):
    """Create one MLflow run for a single diagnostic metric."""
    run_id = run.info.run_id
    rho = run.data.tags.get("rho", "?")
    trial = run.data.tags.get("trial", "?")
    topology = run.data.tags.get("topology", "?")

    tags = {
        "kind": "diagnostics",
        "parent_run_id": run_id,
        "diagnostic_metric": metric_name,
        "rho": rho,
        "trial": trial,
        "topology": topology,
    }

    with mlflow.start_run(
        run_name=f"diag_{metric_name}_{topology}_rho{rho}_t{trial}",
        tags=tags,
    ) as diag_run:
        mlflow.log_params({
            "parent_run_id": run_id,
            "diagnostic_metric": metric_name,
            "split": cfg.pipeline.split,
        })
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Log artifact files for this metric
        if artifact_dir and os.path.isdir(artifact_dir):
            for fname in os.listdir(artifact_dir):
                if fname.startswith(metric_name):
                    fpath = os.path.join(artifact_dir, fname)
                    if os.path.isfile(fpath):
                        mlflow.log_artifact(fpath, artifact_path="diagnostics")

        log_git_info()
        log_resolved_config(cfg)

    metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    print(f"    {metric_name}: {metric_str}  run_id={diag_run.info.run_id}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_mlflow(cfg)

    force = cfg.pipeline.force
    split = cfg.pipeline.split
    cache_dir = get_cache_dir(cfg)
    diag_cfg = cfg.pipeline.diagnostics

    # Determine enabled diagnostics
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

    needs_checkpoint = diag_cfg.weight_norms or diag_cfg.unit_distance_correlation

    exp = mlflow.get_experiment_by_name(cfg.mlflow.experiment_name)
    if exp is None:
        print(f"Experiment '{cfg.mlflow.experiment_name}' not found.")
        return

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.kind = 'model' and attributes.status = 'FINISHED'",
        output_format="list",
    )
    print(f"Found {len(runs)} model runs for diagnostics.")
    print(f"Enabled diagnostics: {enabled}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = get_backend("pt")

    for run in runs:
        run_id = run.info.run_id
        rho = run.data.tags.get("rho", "?")
        trial = run.data.tags.get("trial", "?")

        # Check which metrics still need computing for this model
        needed = []
        for metric_name in enabled:
            if not force:
                existing = _find_finished_diagnostic_run(
                    cfg.mlflow.experiment_name, run_id, metric_name
                )
                if existing is not None:
                    continue
            needed.append(metric_name)

        if not needed:
            print(f"  SKIP rho={rho} trial={trial} (all diagnostics computed)")
            continue

        print(f"  Computing rho={rho} trial={trial}: {needed}")

        diag_dir = os.path.join(artifacts_root, "diagnostics", run_id)
        os.makedirs(diag_dir, exist_ok=True)

        # ── Moran's I ──
        if "morans_i" in needed:
            emb_path = os.path.join(
                artifacts_root, "inference", run_id, split,
                f"embeddings{backend.extension}"
            )
            if backend.exists(emb_path):
                embs = backend.load(emb_path)
                emb_dim = int(run.data.params.get("embedding_dim", 256))
                mi = morans_i(embs, emb_dim)
                _log_diagnostic_run(
                    run, "morans_i", {"morans_i": mi}, diag_dir, cfg,
                )
            else:
                print(f"    WARN: embeddings not cached for {run_id}, skipping morans_i")

        # ── Weight-based diagnostics (share checkpoint loading) ──
        weight_needed = [m for m in needed if m in ("weight_norms", "unit_distance_correlation")]
        if weight_needed:
            encoder, fc_layer = _load_encoder_and_fc(run, device)
            if fc_layer is None:
                print(f"    WARN: no checkpoint for {run_id}, skipping weight diagnostics")
            else:
                import torch.nn as nn
                if not isinstance(fc_layer, nn.Linear):
                    print(f"    WARN: fc_layer is not nn.Linear, skipping weight diagnostics")
                else:
                    if "weight_norms" in weight_needed:
                        wnorms = weight_norms(fc_layer)
                        torch.save(wnorms, os.path.join(diag_dir, "weight_norms.pt"))
                        _log_diagnostic_run(
                            run, "weight_norms",
                            {"weight_norms_mean": float(wnorms.mean()),
                             "weight_norms_std": float(wnorms.std())},
                            diag_dir, cfg,
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
                            run, "unit_distance_correlation", metrics, diag_dir, cfg,
                        )

    print("Diagnostics complete.")


if __name__ == "__main__":
    main()
