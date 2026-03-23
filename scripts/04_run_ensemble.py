#!/usr/bin/env python3
"""
04_run_ensemble.py — Hydra + MLflow ensemble evaluation.

Reads ensemble definitions from ``cfg.ensemble`` (Hydra-composed),
resolves component model runs via declarative selectors, combines logits,
logs results as MLflow behaviour runs.

Supports vote methods (soft, hard, max_confidence, conf_weighted).

HARD FAIL on missing inference artifacts — run 02_cache_inference.py first.

Usage:
    python scripts/04_run_ensemble.py
    python scripts/04_run_ensemble.py ensemble=ce_ensembles
    python scripts/04_run_ensemble.py pipeline.split=val
"""

from __future__ import annotations

import json
import os
import tempfile

import hydra
import mlflow
import mlflow.artifacts
import torch
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.ensemble.combine import combine_logits, METHODS
from src.ensemble.accuracy import ensemble_accuracy, component_accuracies
from src.ensemble.selector import discover_ensembles
from src.data.loaders import get_split_labels
from src.config.paths import get_cache_dir
from src.config.hash import identity_hash
from src.mlflow_utils import (
    behaviour_tags,
    component_set_hash,
    find_finished_ensemble_run,
    log_resolved_config,
    setup_mlflow,
    get_inference_run,
    load_mlflow_artifact,
    safe_to_numpy_float64,
    log_dataset_lineage,
)
from src.mlflow_schema_logger import (
    log_params as schema_log_params,
    start_run as schema_start_run,
    log_tags as schema_log_tags,
)


def _load_inference_artifacts(run_ids, exp_id, split="test"):
    """
    Dynamically fetch MLflow artifacts for each component.
    Returns:
       - logits_list: List of logits arrays across the ensemble
       - labels: Ground truth array
       - composition_map: Dictionary tracking exact inference run ID per model
    """
    logits_list = []
    labels = None
    composition_map = {}

    for i, model_run_id in enumerate(run_ids):
        # 1. Ask MLflow to find the exact inference run for this target model
        inf_runs = get_inference_run([exp_id], model_run_id, split)

        if len(inf_runs) == 0:
            raise RuntimeError(
                f"HARD FAIL: inference run not found for target model {model_run_id} on split '{split}'. "
                f"Please ensure 02_cache_inference.py was executed on the group."
            )

        inf_run_id = inf_runs.iloc[0].run_id

        # 2. Download the tracked tensor artifact
        data = load_mlflow_artifact(
            inf_run_id,
            f"inference_data/{split}_tensors.npz",
            file_type="numpy",
            strict=True,
        )

        # Map predictions to torch format matching old syntax
        logits_tensor = torch.from_numpy(data["logits"])
        logits_list.append(logits_tensor)

        # Map tracker
        composition_map[f"component_{i:03d}"] = {
            "trained_model_run_id": model_run_id,
            "inference_run_id": inf_run_id,
        }

    return logits_list, composition_map


def _run_votes(
    cfg,
    ens_name,
    methods,
    logits_list,
    labels,
    composition_map,
    run_ids,
    cs_hash,
    split_name,
    cache_dir,
    rho_val: str | None = None,
):
    """Run vote-based methods and log identical artifact structures to 02_cache_inference."""
    results = []

    for method in methods:
        if method not in METHODS:
            print(f"  WARN: unknown vote method '{method}', skipping.")
            continue

        step_identity_hash = identity_hash(
            "ensemble",
            component_set_hash=cs_hash,
            split=split_name,
            feature_type="logits",
            method=method,
        )

        # Idempotency
        existing = find_finished_ensemble_run(
            cfg.mlflow.experiment_name, step_identity_hash, ensemble_method=method
        )
        if existing is not None:
            print(
                f"  vote/{method}: already exists (run_id={existing.info.run_id}). Skipping."
            )
            continue

        # Inference Combine (Extract matrix)
        probs = combine_logits(logits_list, method=method)
        acc = ensemble_accuracy(probs, labels)
        preds = torch.argmax(probs, dim=1)

        component_run_ids_csv = ",".join(run_ids)

        tags = behaviour_tags(
            kind="ensemble",
            behaviour=method,
            component_run_ids=run_ids,
            behaviour_input_hash=step_identity_hash,
            component_set_hash=cs_hash,
            rho=rho_val,
            extra={
                "ensemble_name": ens_name,
                "feature_type": "logits",
                "identity_hash": step_identity_hash,
            },
        )
        tags["component_run_ids_csv"] = component_run_ids_csv

        with schema_start_run(
            kind="ensemble", run_name=f"{ens_name}_{method}", tags=tags
        ) as run:

            schema_log_params(
                "ensemble",
                {
                    "method": method,
                    "method_type": "vote",
                    "num_components": len(run_ids),
                    "split": split_name,
                    "rho": rho_val,
                },
            )
            schema_log_tags(
                "ensemble", {"component_run_ids_csv": component_run_ids_csv}
            )

            mlflow.log_metric("ensemble_accuracy", acc)

            comp = component_accuracies(logits_list, labels)
            mlflow.log_metric("comp_mean_acc", comp["mean_acc"])
            mlflow.log_metric("comp_max_acc", comp["max_acc"])

            # ── Link the Component Composition Artifact ──
            with tempfile.TemporaryDirectory() as tmpdir:
                composition_path = os.path.join(tmpdir, "composition_map.json")
                with open(composition_path, "w") as f:
                    json.dump(composition_map, f, indent=4)
                mlflow.log_artifact(composition_path, artifact_path="ensemble_data")

            # ── Ensemble Inference Tracking Parity (Parquet/NPZ) ──
            eval_df = pd.DataFrame(
                {
                    "original_index": safe_to_numpy_float64(torch.arange(len(labels))),
                    "label": safe_to_numpy_float64(labels),
                    "prediction": safe_to_numpy_float64(preds),
                    "confidence": safe_to_numpy_float64(
                        probs.numpy().max(axis=1)
                        if hasattr(probs, "numpy")
                        else probs.max(axis=1)
                    ),
                }
            )

            # Local saves
            os.makedirs(cache_dir, exist_ok=True)
            tabular_path = os.path.join(
                cache_dir, f"{split_name}_{ens_name}_{method}_inference.parquet"
            )
            tensors_path = os.path.join(
                cache_dir, f"{split_name}_{ens_name}_{method}_tensors.npz"
            )

            eval_df.to_parquet(tabular_path, index=False)
            np.savez_compressed(
                tensors_path, probs=probs.numpy() if hasattr(probs, "numpy") else probs
            )

            mlflow.log_artifact(tabular_path, artifact_path="ensemble_data")
            mlflow.log_artifact(tensors_path, artifact_path="ensemble_data")

            log_dataset_lineage(
                labels, split_name, cfg.dataset.name, context="evaluation"
            )

            log_resolved_config(cfg)
            print(
                f"  vote/{method}: acc={acc:.4f} (comp_mean={comp['mean_acc']:.4f})  run_id={run.info.run_id}"
            )

        results.append({"method": method, "acc": acc})
    return results


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_mlflow(cfg)

    split = cfg.pipeline.split
    cache_dir = get_cache_dir(cfg)

    labels = get_split_labels(cfg, split)

    # Get MLflow environment to fetch runs
    exp = mlflow.get_experiment_by_name(cfg.mlflow.experiment_name)
    if exp == None:
        raise ValueError(f"Experiment '{cfg.mlflow.experiment_name}' not found.")

    ensemble_config = OmegaConf.to_container(cfg.ensemble, resolve=True)
    group_by_keys = ensemble_config.get("group_by", ["topology", "rho"])
    min_components = ensemble_config.get("min_components", 2)
    vote_methods = ensemble_config.get(
        "votes", ["soft", "hard", "max_confidence", "conf_weighted"]
    )
    base_filter = ensemble_config.get("filter", {})

    print(f"\n{'='*60}")
    print("Discovering Ensembles...")
    print(f"Grouping keys: {group_by_keys}")

    # Resolve component run IDs dynamically
    try:
        discovered_ensembles = discover_ensembles(
            cfg.mlflow.experiment_name,
            group_by=group_by_keys,
            min_components=min_components,
            base_filter=base_filter,
        )
    except ValueError as e:
        raise RuntimeError(f"HARD FAIL discovering ensembles: {e}")

    if not discovered_ensembles:
        print("No valid ensembles discovered with current filters and min_components.")
        return

    print(f"Discovered {len(discovered_ensembles)} ensemble groups.\n")

    for ens_name, run_ids in discovered_ensembles.items():
        print(f"\n{'='*60}")
        print(f"Executing Ensemble: {ens_name}")
        print(f"  Components matched: {len(run_ids)} runs")

        # Load logits matrices + composition tracking metadata
        logits_list, composition_map = _load_inference_artifacts(
            run_ids, exp.experiment_id, split
        )

        # Determine Rho (unanimous or mixed)
        rhos = set()
        client = mlflow.tracking.MlflowClient()
        for rid in run_ids:
            r = client.get_run(rid)
            r_rho = r.data.params.get("rho")
            if r_rho is not None:
                rhos.add(r_rho)
        rho_sum = rhos.pop() if len(rhos) == 1 else "mixed" if len(rhos) > 1 else None

        # Compute component set hash for idempotency and tagging
        cs_hash = component_set_hash(run_ids)

        # ── Vote methods ──
        if vote_methods:
            _run_votes(
                cfg,
                ens_name,
                vote_methods,
                logits_list,
                labels,
                composition_map,
                run_ids,
                cs_hash,
                split,
                cache_dir,
                rho_val=rho_sum,
            )

    print("\nEnsemble computation complete.")


if __name__ == "__main__":
    main()
