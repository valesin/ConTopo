#!/usr/bin/env python3
"""
04b_compute_diversity.py — Post-ensemble pairwise diversity metrics.

For each ensemble definition, compute prediction-based diversity metrics
(Q-statistic, disagreement, double fault, etc.) across ensemble components.

Reads cached predictions from step 02.

**One MLflow run per (ensemble, metric)**: each diversity metric produces its
own MLflow run (kind=diversity).  Adding a new metric later only computes the
missing one — previously computed metrics are never recalculated.

Usage:
    python scripts/04b_compute_diversity.py
    python scripts/04b_compute_diversity.py pipeline.diversity.metrics='[q_statistic,disagreement]'
    python scripts/04b_compute_diversity.py pipeline.force=true
"""

from __future__ import annotations

import json
import os
import tempfile

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.cache import get_backend
from src.ensemble.selector import resolve_components
from src.config.paths import get_cache_dir
from src.mlflow_utils import (
    component_set_hash,
    log_git_info,
    log_resolved_config,
    setup_mlflow,
)
from src.profiling.diversity import EvalContext, compute_metrics


def _find_finished_diversity_run(
    experiment_name: str, cs_hash: str, metric_name: str, split: str,
):
    """Check if a diversity run already exists for this (ensemble, metric)."""
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return None
    filter_str = (
        f"tags.kind = 'diversity' and "
        f"tags.component_set_hash = '{cs_hash}' and "
        f"tags.diversity_metric = '{metric_name}' and "
        f"tags.split = '{split}' and "
        f"attributes.status = 'FINISHED'"
    )
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=filter_str,
        max_results=1,
        output_format="list",
    )
    return runs[0] if runs else None


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_mlflow(cfg)

    if not cfg.pipeline.diversity.enabled:
        print("Diversity computation disabled (pipeline.diversity.enabled=false).")
        return

    split = cfg.pipeline.split
    force = cfg.pipeline.force
    cache_dir = get_cache_dir(cfg)
    metrics_list = list(cfg.pipeline.diversity.metrics)

    if not metrics_list:
        print("No diversity metrics specified. Nothing to do.")
        return

    ensembles = OmegaConf.to_container(cfg.ensemble.ensembles, resolve=True)
    backend = get_backend("pt")

    for ens_def in ensembles:
        ens_name = ens_def["name"]
        selector = ens_def.get("selector", {})

        print(f"\n{'='*60}")
        print(f"Diversity: {ens_name}")

        # Resolve component run IDs
        try:
            run_ids = resolve_components(selector, cfg.mlflow.experiment_name)
        except ValueError as e:
            print(f"  SKIP: could not resolve ensemble '{ens_name}': {e}")
            continue

        if not run_ids:
            print(f"  SKIP: no component runs for ensemble '{ens_name}'")
            continue

        cs_hash = component_set_hash(run_ids)

        # Check which metrics still need computing
        needed = []
        for metric_name in metrics_list:
            if not force:
                existing = _find_finished_diversity_run(
                    cfg.mlflow.experiment_name, cs_hash, metric_name, split
                )
                if existing is not None:
                    continue
            needed.append(metric_name)

        if not needed:
            print(f"  SKIP: all diversity metrics already computed")
            continue

        print(f"  Components: {len(run_ids)} runs")
        print(f"  Computing: {needed}")

        # Load predictions and labels
        preds_list = []
        labels = None
        logits_list = []
        has_logits = "iou_top_n" in needed

        for run_id in run_ids:
            artifact_dir = str(cache_dir / "inference" / run_id / split)
            preds_path = os.path.join(artifact_dir, f"preds{backend.extension}")
            if not backend.exists(preds_path):
                raise FileNotFoundError(
                    f"HARD FAIL: preds not found at {preds_path}. "
                    f"Run scripts/02_cache_inference.py first."
                )
            preds_list.append(backend.load(preds_path))

            if labels is None:
                labels_path = os.path.join(artifact_dir, f"labels{backend.extension}")
                if backend.exists(labels_path):
                    labels = backend.load(labels_path)

            if has_logits:
                logits_path = os.path.join(artifact_dir, f"logits{backend.extension}")
                if backend.exists(logits_path):
                    logits_list.append(backend.load(logits_path))

        if labels is None:
            raise FileNotFoundError("HARD FAIL: could not find labels for any component run.")

        # Compute all needed metrics at once (they share the EvalContext)
        ctx = EvalContext(
            preds=preds_list,
            labels=labels,
            logits=logits_list if logits_list else None,
        )
        results = compute_metrics(ctx, needed, reduce_group=True)

        # Save locally (all metrics in one file for convenience)
        div_dir = str(cache_dir / "diversity" / ens_name)
        os.makedirs(div_dir, exist_ok=True)

        # Log one MLflow run per metric
        for metric_name, value in results.items():
            # Save individual metric file
            metric_path = os.path.join(div_dir, f"{metric_name}.json")
            with open(metric_path, "w") as f:
                json.dump({metric_name: float(value)}, f, indent=2)

            tags = {
                "kind": "diversity",
                "ensemble_name": ens_name,
                "component_set_hash": cs_hash,
                "diversity_metric": metric_name,
                "split": split,
            }

            with mlflow.start_run(
                run_name=f"div_{ens_name}_{metric_name}",
                tags=tags,
            ) as div_run:
                mlflow.log_params({
                    "ensemble_name": ens_name,
                    "num_components": len(run_ids),
                    "split": split,
                    "diversity_metric": metric_name,
                })
                mlflow.log_metric(metric_name, float(value))

                mlflow.log_artifact(metric_path, artifact_path="diversity")

                # Log component IDs
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                    json.dump({"component_run_ids": run_ids}, f, indent=2)
                    f.flush()
                    mlflow.log_artifact(f.name, artifact_path="diversity")
                    os.unlink(f.name)

                log_git_info()
                log_resolved_config(cfg)

            print(f"    {metric_name}={float(value):.4f}  run_id={div_run.info.run_id}")

    print("\nDiversity computation complete.")


if __name__ == "__main__":
    main()
