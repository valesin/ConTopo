#!/usr/bin/env python3
"""
04_run_ensemble.py — Hydra + MLflow ensemble evaluation.

Reads ensemble definitions from ``cfg.ensemble`` (Hydra-composed),
resolves component model runs via declarative selectors, combines logits,
logs results as MLflow behavior runs.

Supports both vote methods (soft, hard, max_confidence, conf_weighted) and
meta-learner intent registration (meta_lr, meta_mlp — trained in script 05).

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


def _resolve_anchor_selection(meta_def, cfg):
    """Resolve per-meta-learner anchor_selection (falls back to default_anchor_selection)."""
    if "anchor_selection" in meta_def and meta_def["anchor_selection"]:
        return meta_def["anchor_selection"]
    # Fallback to default_anchor_selection from ensemble config
    default_sel = OmegaConf.to_container(cfg.ensemble.default_anchor_selection, resolve=True)
    return default_sel


def _run_meta(cfg, ens_name, meta_defs, logits_list, labels, run_ids, cs_hash, split_name):
    """Register meta-learner intent — actual training is in 05_train_adapters."""
    from src.data.anchors import anchor_spec_hash as compute_anchor_spec_hash

    meta_split_cfg = cfg.ensemble.meta_split
    for meta_def in meta_defs:
        meta_type = meta_def["type"]
        feature_type = meta_def.get("feature_type", "logits")
        hidden_dim = meta_def.get("hidden_dim", None)
        similarity_metric = meta_def.get("similarity_metric", "")

        # Per-meta-learner anchor selection
        anchor_sel = _resolve_anchor_selection(meta_def, cfg)
        # Compute anchor spec hash for this meta-learner's anchor config
        a_spec = {
            "source_split": cfg.pipeline.anchors.source_split,
            "per_class": anchor_sel.get("per_class", 100),
            "strategy": anchor_sel.get("strategy", "per_class_first_n"),
            "order_by": anchor_sel.get("order_by", "example_id"),
            "num_classes": cfg.dataset.num_classes,
        }
        a_spec_hash = compute_anchor_spec_hash(a_spec)

        meta_split_spec = json.dumps({
            "seed": int(meta_split_cfg.seed),
            "strategy": str(meta_split_cfg.strategy),
            "fractions": {
                "train": float(meta_split_cfg.fractions.train),
                "val": float(meta_split_cfg.fractions.val),
                "holdout": float(meta_split_cfg.fractions.holdout),
            },
        }, sort_keys=True)

        bi_hash = behavior_input_hash(
            cs_hash, split=split_name, feature_type=feature_type,
            anchor_spec=a_spec_hash,
            meta_split_spec=meta_split_spec,
            similarity_metric=similarity_metric,
        )

        # Idempotency check
        existing = find_finished_behavior_run(
            cfg.mlflow.experiment_name, bi_hash, behavior=meta_type
        )
        if existing is not None:
            print(f"  meta/{meta_type}: already exists (run_id={existing.info.run_id}). Skipping.")
            continue

        tags = behavior_tags(
            behavior=meta_type,
            component_run_ids=run_ids,
            behavior_input_hash=bi_hash,
            component_set_hash=cs_hash,
            extra={
                "ensemble_name": ens_name,
                "split": split_name,
                "feature_type": feature_type,
                "similarity_metric": similarity_metric,
                "anchor_spec_hash": a_spec_hash,
                "kind": "behavior",
                "meta_type": meta_type,
                "requires_training": "true",
            },
        )

        name_parts = [f"ens_{ens_name}_{meta_type}"]
        if hidden_dim is not None:
            name_parts.append(f"h{hidden_dim}")

        with mlflow.start_run(run_name="_".join(name_parts), tags=tags) as run:
            mlflow.log_params({
                "ensemble_name": ens_name,
                "method": meta_type,
                "method_type": "meta",
                "feature_type": feature_type,
                "similarity_metric": similarity_metric,
                "anchor_spec_hash": a_spec_hash,
                "num_components": len(run_ids),
                "split": split_name,
                "meta_split_seed": int(meta_split_cfg.seed),
                "meta_split_strategy": str(meta_split_cfg.strategy),
                "meta_split_train_frac": float(meta_split_cfg.fractions.train),
                "meta_split_val_frac": float(meta_split_cfg.fractions.val),
                "meta_split_holdout_frac": float(meta_split_cfg.fractions.holdout),
            })
            if hidden_dim is not None:
                mlflow.log_params({"hidden_dim": hidden_dim})

            log_git_info()
            log_resolved_config(cfg)
            print(f"  meta/{meta_type}: registered (train in script 05)  run_id={run.info.run_id}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_mlflow(cfg)

    split = cfg.pipeline.split
    artifacts_root = cfg.runtime.artifacts_root

    # Load manifest for compatibility verification
    manifest = get_or_create_manifest(
        dataset_name=cfg.dataset.name,
        split=split,
        data_root=cfg.runtime.data_root,
        artifacts_root=artifacts_root,
    )
    manifest_hash = manifest.manifest_hash

    # Ensemble config from Hydra (no yaml.safe_load)
    ensembles = OmegaConf.to_container(cfg.ensemble.ensembles, resolve=True)

    for ens_def in ensembles:
        ens_name = ens_def["name"]
        selector = ens_def.get("selector", {})
        vote_methods = ens_def.get("votes", ["soft", "hard", "max_confidence", "conf_weighted"])
        meta_defs = ens_def.get("meta", [])

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
        logits_list, labels = _load_logits_for_runs(run_ids, artifacts_root, split)

        # Compute hashes
        cs_hash = component_set_hash(run_ids)
        bi_hash = behavior_input_hash(cs_hash, split=split, feature_type="logits")

        # ── Vote methods ──
        if vote_methods:
            _run_votes(cfg, ens_name, vote_methods, logits_list, labels, run_ids, cs_hash, bi_hash, split)

        # ── Meta-learner methods (register intent) ──
        if meta_defs:
            _run_meta(cfg, ens_name, meta_defs, logits_list, labels, run_ids, cs_hash, split)

    print("\nEnsemble computation complete.")


if __name__ == "__main__":
    main()
