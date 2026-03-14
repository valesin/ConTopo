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
from src.data.manifest import get_or_create_manifest
from src.config.paths import get_cache_dir
from src.mlflow_utils import (
    behaviour_tags,
    component_set_hash,
    behaviour_input_hash,
    find_finished_ensemble_run,
    log_resolved_config,
    setup_mlflow,
    get_inference_run,
    load_mlflow_artifact,
    log_manifest_lineage,
    safe_to_numpy_float64,
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
        data = load_mlflow_artifact(inf_run_id, f"inference_data/{split}_tensors.npz", file_type="numpy", strict=True)

        # Map predictions to torch format matching old syntax
        logits_tensor = torch.from_numpy(data["logits"])
        logits_list.append(logits_tensor)

        # Map tracker
        composition_map[f"component_{i:03d}"] = {
            "trained_model_run_id": model_run_id,
            "inference_run_id": inf_run_id,
        }

    return logits_list, composition_map


def _verify_manifest_compat(run_ids, expected_hash):
    """Verify all component runs share the same dataset_manifest_hash."""
    client = mlflow.tracking.MlflowClient()
    mismatched = []
    for run_id in run_ids:
        run = client.get_run(run_id)
        run_hash = run.data.tags.get("dataset_manifest_hash", "")
        if run_hash and run_hash != expected_hash:
            mismatched.append((run_id, run_hash))
    if mismatched:
        raise ValueError(
            f"Manifest hash mismatch! Expected {expected_hash}, but "
            f"{len(mismatched)} runs differ: {mismatched[:3]}..."
        )


def _run_votes(
    cfg,
    ens_name,
    methods,
    logits_list,
    manifest,
    composition_map,
    run_ids,
    cs_hash,
    bi_hash,
    split_name,
    cache_dir,
    rho_val: str | None = None,
):
    """Run vote-based methods and log identical artifact structures to 02_cache_inference."""
    results = []

    labels = manifest.labels

    for method in methods:
        if method not in METHODS:
            print(f"  WARN: unknown vote method '{method}', skipping.")
            continue

        # Idempotency
        existing = find_finished_ensemble_run(
            cfg.mlflow.experiment_name, bi_hash, ensemble_method=method
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

        tags = behaviour_tags(
            kind="ensemble",
            behaviour=method,
            component_run_ids=run_ids,
            behaviour_input_hash=bi_hash,
            component_set_hash=cs_hash,
            rho=rho_val,
            extra={
                "ensemble_name": ens_name,
                "split": split_name,
                "feature_type": "logits",
                "dataset_manifest_hash": manifest.manifest_hash,
            },
        )

        with mlflow.start_run(run_name=f"{ens_name}_{method}", tags=tags) as run:

            mlflow.log_params(
                {
                    "ensemble_name": ens_name,
                    "method": method,
                    "method_type": "vote",
                    "num_components": len(run_ids),
                    "split": split_name,
                    "dataset_manifest_hash": manifest.manifest_hash,
                }
            )

            mlflow.log_metric("ensemble_accuracy", acc)

            comp = component_accuracies(logits_list, labels)
            mlflow.log_metric("comp_mean_acc", comp["mean_acc"])
            mlflow.log_metric("comp_max_acc", comp["max_acc"])

            # ── Link the Component Composition Artifact ──
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(composition_map, f, indent=4)
                f.flush()
                mlflow.log_artifact(f.name, artifact_path="ensemble_data")
                os.unlink(f.name)

            # ── Ensemble Inference Tracking Parity (Parquet/NPZ) ──
            eval_df = pd.DataFrame(
                {
                    "example_id": manifest.hashes,
                    "original_index": safe_to_numpy_float64(manifest.original_indices),
                    "label": safe_to_numpy_float64(manifest.labels),
                    "prediction": safe_to_numpy_float64(preds),
                    "confidence": safe_to_numpy_float64(
                        probs.numpy().max(axis=1) if hasattr(probs, "numpy") else probs.max(axis=1)
                    ),
                }
            )

            # Dataset tracking for data lineage
            log_manifest_lineage(manifest, split_name, cfg.dataset.name, context="validation" if split_name == "val" else "testing")

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

    manifest = get_or_create_manifest(
        dataset_name=cfg.dataset.name,
        split=split,
        data_root=cfg.runtime.data_root,
        artifacts_root=str(cache_dir),
    )
    manifest_hash = manifest.manifest_hash

    # Get MLflow environment to fetch runs
    exp = mlflow.get_experiment_by_name(cfg.mlflow.experiment_name)
    if exp == None:
        raise ValueError(f"Experiment '{cfg.mlflow.experiment_name}' not found.")

    ensemble_config = OmegaConf.to_container(cfg.ensemble, resolve=True)
    group_by_keys = ensemble_config.get("group_by", ["loss_type", "topology", "rho"])
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

        # Verify manifest compatibility
        _verify_manifest_compat(run_ids, manifest_hash)

        # Load logits matrices + composition tracking metadata
        logits_list, composition_map = _load_inference_artifacts(
            run_ids, exp.experiment_id, split
        )

        # Determine Rho (unanimous or mixed)
        rhos = set()
        client = mlflow.tracking.MlflowClient()
        for rid in run_ids:
            r = client.get_run(rid)
            r_rho = r.data.tags.get("rho")
            if r_rho is not None:
                rhos.add(r_rho)
        rho_sum = rhos.pop() if len(rhos) == 1 else "mixed" if len(rhos) > 1 else None

        # Compute hashes for idempotency and tagging
        cs_hash = component_set_hash(run_ids)
        bi_hash = behaviour_input_hash(
            component_set_hash_val=cs_hash,
            dataset_manifest_hash=manifest_hash,
            split=split,
            feature_type="logits",
        )

        # ── Vote methods ──
        if vote_methods:
            _run_votes(
                cfg,
                ens_name,
                vote_methods,
                logits_list,
                manifest,
                composition_map,
                run_ids,
                cs_hash,
                bi_hash,
                split,
                cache_dir,
                rho_val=rho_sum,
            )

    print("\nEnsemble computation complete.")


if __name__ == "__main__":
    main()
