#!/usr/bin/env python3
"""
04c_compute_consistency.py — Post-ensemble RDM/RSA consistency.

For each ensemble definition, compute per-model RDMs on anchor embeddings
and pairwise RSA correlation between all models.  Measures representational
consistency within the ensemble.

Reads cached embeddings from step 02 and anchor selection from pipeline config.

Results are logged as MLflow runs (kind=consistency), tagged with
ensemble_name and component run IDs for easy link-back.

Usage:
    python scripts/04c_compute_consistency.py
    python scripts/04c_compute_consistency.py pipeline.force=true
    python scripts/04c_compute_consistency.py pipeline.consistency.enabled=false
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile

import hydra
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.anchors import get_or_create_anchors
from src.data.cache import get_backend
from src.data.manifest import get_or_create_manifest
from src.ensemble.selector import resolve_components
from src.config.paths import get_cache_dir
from src.mlflow_utils import (
    component_set_hash,
    log_git_info,
    log_resolved_config,
    setup_mlflow,
)
from src.profiling.rdm import pearson_rdm, rsa_correlation


def _consistency_hash(cs_hash: str, anchor_spec_hash: str, split: str) -> str:
    """Idempotency hash for a consistency computation."""
    parts = [cs_hash, anchor_spec_hash, split, "consistency"]
    canonical = json.dumps(parts, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _find_finished_consistency_run(experiment_name: str, cons_hash: str):
    """Check if a consistency run already exists for this hash."""
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return None
    filter_str = (
        f"tags.kind = 'consistency' and "
        f"tags.consistency_hash = '{cons_hash}' and "
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

    if not cfg.pipeline.consistency.enabled:
        print("Consistency computation disabled (pipeline.consistency.enabled=false).")
        return

    split = cfg.pipeline.split
    force = cfg.pipeline.force
    cache_dir = get_cache_dir(cfg)
    anchors_cfg = cfg.pipeline.anchors

    # Manifest + anchors
    manifest = get_or_create_manifest(
        dataset_name=cfg.dataset.name,
        split=split,
        data_root=cfg.runtime.data_root,
        artifacts_root=str(cache_dir),
    )
    anchors = get_or_create_anchors(
        manifest,
        per_class=anchors_cfg.per_class,
        strategy=anchors_cfg.strategy,
        order_by=anchors_cfg.order_by,
        artifacts_root=str(cache_dir),
    )
    anchor_indices = anchors["anchor_indices"]
    a_spec_hash = anchors["spec_hash"]

    ensembles = OmegaConf.to_container(cfg.ensemble.ensembles, resolve=True)
    backend = get_backend("pt")

    for ens_def in ensembles:
        ens_name = ens_def["name"]
        selector = ens_def.get("selector", {})

        print(f"\n{'='*60}")
        print(f"Consistency: {ens_name}")

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
        cons_hash = _consistency_hash(cs_hash, a_spec_hash, split)

        # Idempotency
        if not force:
            existing = _find_finished_consistency_run(
                cfg.mlflow.experiment_name, cons_hash
            )
            if existing is not None:
                print(f"  SKIP: already computed (run_id={existing.info.run_id})")
                continue

        print(f"  Components: {len(run_ids)} runs")

        # Compute per-model RDMs from anchor embeddings
        cons_dir = str(cache_dir / "consistency" / ens_name)
        os.makedirs(cons_dir, exist_ok=True)

        rdms = {}
        skip = False
        for run_id in run_ids:
            emb_path = str(
                cache_dir / "inference" / run_id / split /
                f"embeddings{backend.extension}"
            )
            if not backend.exists(emb_path):
                print(f"  SKIP: embeddings not cached for {run_id}")
                skip = True
                break

            embeddings = backend.load(emb_path)
            anchor_embs = embeddings[anchor_indices]  # [K, D]
            rdm = pearson_rdm(anchor_embs)  # [K, K]
            rdms[run_id] = rdm

            # Save individual RDM
            torch.save(rdm, os.path.join(cons_dir, f"{run_id}_rdm.pt"))

        if skip:
            continue

        # Compute pairwise RSA correlation matrix
        n_models = len(run_ids)
        rsa_matrix = np.zeros((n_models, n_models))
        for i, rid_i in enumerate(run_ids):
            for j, rid_j in enumerate(run_ids):
                if i == j:
                    rsa_matrix[i, j] = 1.0
                elif j > i:
                    r = rsa_correlation(rdms[rid_i], rdms[rid_j])
                    rsa_matrix[i, j] = r
                    rsa_matrix[j, i] = r

        # Mean off-diagonal RSA
        if n_models >= 2:
            mask = ~np.eye(n_models, dtype=bool)
            mean_rsa = float(np.mean(rsa_matrix[mask]))
        else:
            mean_rsa = float("nan")

        # Save RSA matrix
        rsa_path = os.path.join(cons_dir, "rsa_matrix.pt")
        torch.save(torch.tensor(rsa_matrix, dtype=torch.float32), rsa_path)

        # Save run ID ordering for the matrix
        ordering_path = os.path.join(cons_dir, "run_id_ordering.json")
        with open(ordering_path, "w") as f:
            json.dump(run_ids, f, indent=2)

        # Log MLflow run
        tags = {
            "kind": "consistency",
            "ensemble_name": ens_name,
            "component_set_hash": cs_hash,
            "consistency_hash": cons_hash,
            "anchor_spec_hash": a_spec_hash,
        }

        with mlflow.start_run(
            run_name=f"cons_{ens_name}",
            tags=tags,
        ) as cons_run:
            mlflow.log_params({
                "ensemble_name": ens_name,
                "num_components": n_models,
                "split": split,
                "anchor_spec_hash": a_spec_hash,
                "anchors_per_class": anchors_cfg.per_class,
            })
            mlflow.log_metric("mean_rsa_correlation", mean_rsa)

            # Log all artifacts in the consistency dir
            for fname in os.listdir(cons_dir):
                fpath = os.path.join(cons_dir, fname)
                if os.path.isfile(fpath):
                    mlflow.log_artifact(fpath, artifact_path="consistency")

            log_git_info()
            log_resolved_config(cfg)

        print(f"  Done. mean_rsa={mean_rsa:.4f}  run_id={cons_run.info.run_id}")

    print("\nConsistency computation complete.")


if __name__ == "__main__":
    main()
