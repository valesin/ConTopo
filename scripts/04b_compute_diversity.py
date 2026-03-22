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

import json
import os
import tempfile

import hydra
import mlflow
import torch
from omegaconf import DictConfig

from src.data.loaders import get_split_labels
from src.ensemble.selector import discover_ensembles
from src.config.paths import get_cache_dir
from src.mlflow_utils import (
    log_resolved_config,
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

    # 1. Get ground-truth labels
    labels = get_split_labels(cfg, split)

    # 2. Discover ensemble groups dynamically from the actual DB tracking
    groups = discover_ensembles(cfg.mlflow.experiment_name)

    print(f"\nDiscovered {len(groups)} ensemble groups from MLflow.")

    for ens_name, run_ids in groups.items():
        print(f"\n{'='*60}")
        print(f"Diversity: {ens_name}")

        cs_hash = component_set_hash(run_ids)

        # Check which metrics still need computing
        needed = []
        for metric_name in metrics_list:
            if not force:
                existing = find_finished_diversity_run(
                    cfg.mlflow.experiment_name, cs_hash, metric_name, split
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
                f"inference_data/{split}_inference_results.parquet",
                file_type="parquet",
                strict=True,
            )
            preds_list.append(torch.tensor(df["prediction"].values))

            if has_logits:
                data = load_mlflow_artifact(
                    inf_run_id,
                    f"inference_data/{split}_tensors.npz",
                    file_type="numpy",
                    strict=True,
                )
                logits_list.append(torch.from_numpy(data["logits"]))

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
                "ensemble_name": ens_name,
                "component_set_hash": cs_hash,
                "run_name": f"{ens_name}_{metric_name}",
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

                mlflow.log_artifact(metric_path, artifact_path="diversity")

                # Log component IDs
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as f:
                    json.dump({"component_run_ids": run_ids}, f, indent=2)
                    f.flush()
                    mlflow.log_artifact(f.name, artifact_path="diversity")
                    os.unlink(f.name)

                log_dataset_lineage(
                    labels, split, cfg.dataset.name, context="evaluation"
                )

                log_resolved_config(cfg)

            print(f"    {metric_name}={float(value):.4f}  run_id={div_run.info.run_id}")

    print("\nDiversity computation complete.")


if __name__ == "__main__":
    main()
