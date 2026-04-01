#!/usr/bin/env python3
"""
04b_compute_diversity.py — Post-ensemble pairwise diversity metrics.

For each ensemble definition, compute prediction-based diversity metrics
(Q-statistic, disagreement, double fault, etc.) across ensemble components.
Reads cached predictions from step 02 via MLflow Artifact tracking.

**One MLflow run per (ensemble, metric)**: each diversity metric produces its
own MLflow run (kind=diversity). Adding a new metric later only computes the
missing one — previously computed metrics are never recalculated.
"""

from __future__ import annotations

import os

import hydra
import mlflow
import torch
from omegaconf import DictConfig

from src.data.loaders import get_split_labels
from src.ensemble.selector import discover_ensembles_from_cfg
from src.config.hash import identity_hash
from src.mlflow_utils import (
    setup_mlflow,
    get_inference_run,
    load_mlflow_artifact,
    component_set_hash,
    find_finished_diversity_run,
    log_dataset_lineage,
)
from src.profiling.diversity import EvalContext, compute_metrics
from src.mlflow_schema_logger import (
    log_params as schema_log_params,
    start_run as schema_start_run,
)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_mlflow(cfg)

    if not cfg.analysis.diversity.enabled:
        print("Diversity computation disabled (analysis.diversity.enabled=false).")
        return

    split = cfg.execution.split
    force = cfg.execution.force
    metrics_list = list(cfg.analysis.diversity.metrics)

    if not metrics_list:
        print("No diversity metrics specified. Nothing to do.")
        return

    # 1. Get ground-truth labels
    labels = get_split_labels(cfg, split)

    # 2. Discover ensemble groups dynamically from the actual DB tracking
    groups = discover_ensembles_from_cfg(cfg, cfg.mlflow.experiment_name)

    print(f"\nDiscovered {len(groups)} ensemble groups from MLflow.")

    for ens_name, run_ids in groups.items():
        print(f"\n{'='*60}")
        print(f"Diversity: {ens_name}")

        cs_hash = component_set_hash(run_ids)

        # Check which metrics still need computing
        needed = []
        for metric_name in metrics_list:
            if not force:
                step_identity_hash = identity_hash(
                    "diversity",
                    component_set_hash=cs_hash,
                    diversity_metric=metric_name,
                    split=split,
                )
                existing = find_finished_diversity_run(
                    cfg.mlflow.experiment_name, step_identity_hash
                )
                if existing is not None:
                    continue
            needed.append(metric_name)

        if not needed:
            print("  SKIP: all diversity metrics already computed")
            continue

        print(f"  Components: {len(run_ids)} runs")
        print(f"  Computing: {needed}")

        # Load predictions and logits
        preds_list = []
        logits_list = []
        has_logits = "iou_top_n" in needed

        for i, run_id in enumerate(run_ids):
            # Find the corresponding inference run
            inf_runs = get_inference_run(
                [
                    mlflow.get_experiment_by_name(
                        cfg.mlflow.experiment_name
                    ).experiment_id
                ],
                run_id,
                split,
            )

            if inf_runs.empty:
                raise RuntimeError(
                    f"HARD FAIL: Could not find FINISHED '{split}' inference run for "
                    f"model {run_id}. Run scripts/02_cache_inference.py first."
                )

            inf_run_id = inf_runs.iloc[0].run_id

            # We can download predictions efficiently via tabular tracking
            df = load_mlflow_artifact(
                inf_run_id,
                f"inference/{split}_inference_results.parquet",
                file_type="parquet",
                strict=True,
                cache_dir=cfg.mlflow.artifact_cache_dir,
            )
            preds_list.append(torch.tensor(df["prediction"].values))

            if has_logits:
                data = load_mlflow_artifact(
                    inf_run_id,
                    f"inference/{split}_tensors.npz",
                    file_type="numpy",
                    strict=True,
                    cache_dir=cfg.mlflow.artifact_cache_dir,
                )
                logits_list.append(torch.from_numpy(data["logits"]))

        # Compute all needed metrics at once (they share the EvalContext)
        ctx = EvalContext(
            preds=preds_list,
            labels=labels,
            logits=logits_list if logits_list else None,
        )
        results = compute_metrics(ctx, needed, reduce_group=True)

        # Log one MLflow run per metric
        for metric_name, value in results.items():
            step_identity_hash = identity_hash(
                "diversity",
                component_set_hash=cs_hash,
                diversity_metric=metric_name,
                split=split,
            )

            component_run_ids_csv = ",".join(run_ids)

            tags = {
                "ensemble_name": ens_name,
                "component_set_hash": cs_hash,
                "identity_hash": step_identity_hash,
                "run_name": f"{ens_name}_{metric_name}",
                "component_run_ids_csv": component_run_ids_csv,
            }

            with schema_start_run(
                kind="diversity",
                run_name=f"{ens_name}_{metric_name}",
                tags=tags,
            ) as div_run:
                schema_log_params(
                    "diversity",
                    {
                        "num_components": len(run_ids),
                        "split": split,
                        "diversity_metric": metric_name,
                    },
                )
                mlflow.log_metric(metric_name, float(value))

                log_dataset_lineage(
                    labels, split, cfg.dataset.name, context="evaluation"
                )

            print(f"    {metric_name}={float(value):.4f}  run_id={div_run.info.run_id}")

    print("\nDiversity computation complete.")


if __name__ == "__main__":
    main()
