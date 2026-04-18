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
    python scripts/04_run_ensemble.py execution.split=val
"""

from __future__ import annotations

import json
import os
import tempfile

import hydra
import torch
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.ensemble.combine import combine_logits, METHODS
from src.ensemble.accuracy import ensemble_accuracy, component_accuracies
from src.ensemble.selector import discover_ensembles_from_cfg, encode_groups_signature
from src.data.loaders import get_split_labels
from src.config.hash import identity_hash
from src.mlflow_utils import (
    apply_mlflow_env_overrides,
    behaviour_tags,
    component_set_hash,
    setup_mlflow,
    load_mlflow_artifact,
    get_run_context,
    safe_to_numpy_float64,
    log_dataset_lineage,
)
from src.repositories.functional_run_repository import (
    configure_run_repository,
    find_finished_identity_run,
    get_run,
)
from src.mlflow_schema_logger import (
    log_params as schema_log_params,
    start_run as schema_start_run,
    log_tags as schema_log_tags,
    timed_log_metric,
    timed_log_artifact,
)


def _load_inference_artifacts(cfg, run_ids, split="test"):
    """
    Dynamically fetch MLflow artifacts for each component.
    Returns:
       - logits_list: List of logits tensors across the ensemble
       - composition_map: Dictionary tracking exact inference run ID per model
    """
    logits_list = []
    composition_map = {}

    for i, model_run_id in enumerate(run_ids):
        # 1. Ask MLflow to find the exact inference run for this target model
        inf_identity = identity_hash(
            "inference", trained_model_run_id=model_run_id, split=split
        )
        inf_run = find_finished_identity_run("inference", inf_identity)

        if inf_run is None:
            raise RuntimeError(
                f"HARD FAIL: inference run not found for target model {model_run_id} on split '{split}'. "
                f"Please ensure 02_cache_inference.py was executed on the group."
            )

        inf_run_id = inf_run.info.run_id

        # 2. Download the tracked tensor artifact
        data = load_mlflow_artifact(
            inf_run_id,
            f"inference/{split}_tensors.npz",
            file_type="numpy",
            strict=True,
            cache_dir=cfg.mlflow.artifact_cache_dir,
        )

        # Map predictions to torch format matching old syntax
        logits_tensor = torch.from_numpy(data["logits"])
        logits_list.append(logits_tensor)

        print(
            f"  Loaded inference artifacts for component {i:03d} (model={model_run_id[:8]}… inf={inf_run_id[:8]}…)"
        )

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
    rho_val: str | None = None,
    groups_sig: str | None = None,
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
        existing = find_finished_identity_run("ensemble", step_identity_hash)
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

        extra: dict = {
            "ensemble_name": ens_name,
            "feature_type": "logits",
            "identity_hash": step_identity_hash,
        }
        if groups_sig is not None:
            extra["groups_signature"] = groups_sig

        tags = behaviour_tags(
            kind="ensemble",
            behaviour=method,
            component_run_ids=run_ids,
            behaviour_input_hash=step_identity_hash,
            component_set_hash=cs_hash,
            rho=rho_val,
            extra=extra,
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

            timed_log_metric("ensemble_accuracy", acc)

            comp = component_accuracies(logits_list, labels)
            timed_log_metric("comp_mean_acc", comp["mean_acc"])
            timed_log_metric("comp_max_acc", comp["max_acc"])

            # ── Link the Component Composition Artifact ──
            with tempfile.TemporaryDirectory() as tmpdir:
                composition_path = os.path.join(tmpdir, "composition_map.json")
                with open(composition_path, "w") as f:
                    json.dump(composition_map, f, indent=4)
                timed_log_artifact(composition_path, artifact_path="ensemble")

            # ── Ensemble Inference Tracking Parity (Parquet/NPZ) ──
            eval_df = pd.DataFrame(
                {
                    "original_index": safe_to_numpy_float64(torch.arange(len(labels))),
                    "label": safe_to_numpy_float64(labels),
                    "prediction": safe_to_numpy_float64(preds),
                }
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                tabular_path = os.path.join(
                    tmpdir, f"{split_name}_{ens_name}_{method}_inference.parquet"
                )
                tensors_path = os.path.join(
                    tmpdir, f"{split_name}_{ens_name}_{method}_tensors.npz"
                )

                eval_df.to_parquet(tabular_path, index=False)
                np.savez_compressed(
                    tensors_path,
                    probs=probs.numpy() if hasattr(probs, "numpy") else probs,
                )

                timed_log_artifact(tabular_path, artifact_path="ensemble")
                timed_log_artifact(tensors_path, artifact_path="ensemble")

            log_dataset_lineage(
                labels, split_name, cfg.dataset.name, context="evaluation"
            )

            print(
                f"  vote/{method}: acc={acc:.4f} (comp_mean={comp['mean_acc']:.4f})  run_id={run.info.run_id}"
            )

        results.append({"method": method, "acc": acc})
    return results


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    apply_mlflow_env_overrides(cfg)
    setup_mlflow(cfg)
    configure_run_repository(cfg.mlflow.tracking_uri, cfg.mlflow.experiment_name)

    split = cfg.execution.split
    groups_sig = encode_groups_signature(cfg.groups)

    labels = get_split_labels(cfg, split)

    vote_methods = list(cfg.ensemble.votes)

    print(f"\n{'='*60}")
    print("Discovering Ensembles...")
    print(f"Grouping keys: {list(cfg.groups.group_by)}")

    # Resolve component run IDs dynamically
    try:
        discovered_ensembles = discover_ensembles_from_cfg(
            cfg, cfg.mlflow.experiment_name
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

        # Compute component set hash first (pure computation, no I/O)
        cs_hash = component_set_hash(run_ids)

        # ── Idempotency pre-check: skip artifact loading if all methods already cached ──
        if vote_methods:
            uncached_methods = [
                m
                for m in vote_methods
                if find_finished_identity_run(
                    "ensemble",
                    identity_hash(
                        "ensemble",
                        component_set_hash=cs_hash,
                        split=split,
                        feature_type="logits",
                        method=m,
                    ),
                )
                is None
            ]
            if not uncached_methods:
                print(f"  All vote methods already cached. Skipping.")
                continue
        else:
            uncached_methods = []

        # Load logits matrices + composition tracking metadata
        logits_list, composition_map = _load_inference_artifacts(cfg, run_ids, split)

        # Determine Rho (unanimous or mixed)
        rhos = set()
        for rid in run_ids:
            r = get_run(rid)
            r_rho, _, _ = get_run_context(r)
            if r_rho != "?":
                rhos.add(r_rho)
        rho_sum = rhos.pop() if len(rhos) == 1 else "mixed" if len(rhos) > 1 else None

        # ── Vote methods ──
        if uncached_methods:
            _run_votes(
                cfg,
                ens_name,
                uncached_methods,
                logits_list,
                labels,
                composition_map,
                run_ids,
                cs_hash,
                split,
                rho_val=rho_sum,
                groups_sig=groups_sig,
            )

    print("\nEnsemble computation complete.")


if __name__ == "__main__":
    main()
