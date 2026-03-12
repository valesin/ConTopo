#!/usr/bin/env python3
"""
04_run_ensemble.py — Hydra + MLflow ensemble evaluation.

Reads ensemble definitions from ``cfg.ensemble`` (Hydra-composed),
resolves component model runs via declarative selectors, combines logits,
logs results as MLflow behavior runs.

Supports vote methods (soft, hard, max_confidence, conf_weighted).

HARD FAILURE on missing inference artifacts — run 02_cache_inference.py first.

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
import torch
from omegaconf import DictConfig, OmegaConf

from src.ensemble.combine import combine_logits, METHODS
from src.ensemble.accuracy import ensemble_accuracy, component_accuracies
from src.ensemble.selector import resolve_components
from src.data.cache import get_backend
from src.data.manifest import get_or_create_manifest
from src.config.paths import get_cache_dir
from src.mlflow_utils import (
    behavior_tags,
    component_set_hash,
    behavior_input_hash,
    find_finished_behavior_run,
    log_git_info,
    log_resolved_config,
    setup_mlflow,
)


def _load_logits_for_runs(run_ids, artifacts_root, split="test"):
    """Load logits artifacts for a list of run_ids. HARD FAIL on missing."""
    backend = get_backend("pt")
    logits_list = []
    labels = None

    for run_id in run_ids:
        artifact_dir = os.path.join(artifacts_root, "inference", run_id, split)
        logits_path = os.path.join(artifact_dir, f"logits{backend.extension}")

        if not backend.exists(logits_path):
            raise FileNotFoundError(
                f"HARD FAIL: logits not found for run {run_id} at {logits_path}. "
                f"Run scripts/02_cache_inference.py first."
            )

        logits_list.append(backend.load(logits_path))

        if labels is None:
            labels_path = os.path.join(artifact_dir, f"labels{backend.extension}")
            if backend.exists(labels_path):
                labels = backend.load(labels_path)

    if labels is None:
        raise FileNotFoundError("HARD FAIL: could not find labels for any component run.")

    return logits_list, labels


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
    cfg, ens_name, methods, logits_list, labels, run_ids, cs_hash, bi_hash, split_name
):
    """Run simple vote-based ensemble methods with idempotency."""
    results = []
    for method in methods:
        if method not in METHODS:
            print(f"  WARN: unknown vote method '{method}', skipping.")
            continue

        # Idempotency check
        existing = find_finished_behavior_run(
            cfg.mlflow.experiment_name, bi_hash, behavior=method
        )
        if existing is not None:
            print(f"  vote/{method}: already exists (run_id={existing.info.run_id}). Skipping.")
            continue

        probs = combine_logits(logits_list, method=method)
        acc = ensemble_accuracy(probs, labels)

        tags = behavior_tags(
            behavior=method,
            component_run_ids=run_ids,
            behavior_input_hash=bi_hash,
            component_set_hash=cs_hash,
            extra={
                "ensemble_name": ens_name,
                "split": split_name,
                "feature_type": "logits",
                "kind": "behavior",
            },
        )

        with mlflow.start_run(run_name=f"ens_{ens_name}_{method}", tags=tags) as run:
            mlflow.log_params({
                "ensemble_name": ens_name,
                "method": method,
                "method_type": "vote",
                "num_components": len(run_ids),
                "split": split_name,
            })
            mlflow.log_metric("ensemble_accuracy", acc)

            comp = component_accuracies(logits_list, labels)
            mlflow.log_metric("comp_mean_acc", comp["mean_acc"])
            mlflow.log_metric("comp_max_acc", comp["max_acc"])

            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump({"component_run_ids": run_ids}, f, indent=2)
                f.flush()
                mlflow.log_artifact(f.name, artifact_path="ensemble")
                os.unlink(f.name)

            log_git_info()
            log_resolved_config(cfg)
            print(f"  vote/{method}: acc={acc:.4f} (comp_mean={comp['mean_acc']:.4f})  run_id={run.info.run_id}")

        results.append({"method": method, "acc": acc})
    return results


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_mlflow(cfg)

    split = cfg.pipeline.split
    cache_dir = get_cache_dir(cfg)

    # Load manifest for compatibility verification
    manifest = get_or_create_manifest(
        dataset_name=cfg.dataset.name,
        split=split,
        data_root=cfg.runtime.data_root,
        artifacts_root=str(cache_dir),
    )
    manifest_hash = manifest.manifest_hash

    # Ensemble config from Hydra (no yaml.safe_load)
    ensembles = OmegaConf.to_container(cfg.ensemble.ensembles, resolve=True)

    for ens_def in ensembles:
        ens_name = ens_def["name"]
        selector = ens_def.get("selector", {})
        vote_methods = ens_def.get("votes", ["soft", "hard", "max_confidence", "conf_weighted"])

        print(f"\n{'='*60}")
        print(f"Ensemble: {ens_name}")
        print(f"Selector: {selector}")

        # Resolve component run IDs
        try:
            run_ids = resolve_components(selector, cfg.mlflow.experiment_name)
        except ValueError as e:
            raise RuntimeError(f"HARD FAIL resolving ensemble '{ens_name}': {e}")

        if not run_ids:
            raise RuntimeError(f"HARD FAIL: no component runs found for ensemble '{ens_name}'")

        print(f"  Components: {len(run_ids)} runs")

        # Verify manifest compatibility
        _verify_manifest_compat(run_ids, manifest_hash)

        # Load logits — HARD FAIL on missing
        logits_list, labels = _load_logits_for_runs(run_ids, str(cache_dir), split)

        # Compute hashes
        cs_hash = component_set_hash(run_ids)
        bi_hash = behavior_input_hash(cs_hash, split=split, feature_type="logits")

        # ── Vote methods ──
        if vote_methods:
            _run_votes(cfg, ens_name, vote_methods, logits_list, labels, run_ids, cs_hash, bi_hash, split)

    print("\nEnsemble computation complete.")


if __name__ == "__main__":
    main()
