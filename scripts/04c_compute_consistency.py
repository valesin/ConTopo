#!/usr/bin/env python3
"""
04c_compute_consistency.py — Post-ensemble RDM/RSA consistency.

For each ensemble definition, compute per-model RDMs on anchor embeddings
and pairwise RSA correlation between all models. Measures representational
consistency within the ensemble.

Reads cached embeddings from tracked MLflow Artifacts and anchor selection from
pipeline config.

Results are logged as MLflow runs (kind=consistency), tagged with
ensemble_name and component run IDs for easy link-back.
"""

from __future__ import annotations

import json
import os
import tempfile

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.data.anchors import get_or_create_anchors
from src.data.loaders import get_num_classes, get_split_labels
from src.ensemble.selector import discover_ensembles_from_cfg
from src.config.paths import get_anchors_dir
from src.config.hash import identity_hash
from src.mlflow_utils import (
    apply_mlflow_env_overrides,
    component_set_hash,
    setup_mlflow,
    load_mlflow_artifact,
    log_dataset_lineage,
)
from src.repositories.functional_run_repository import (
    configure_run_repository,
    find_finished_identity_run,
)
from src.profiling.rdm import pearson_rdm, rsa_correlation
from src.mlflow_schema_logger import (
    log_params as schema_log_params,
    start_run as schema_start_run,
    timed_log_metric,
    timed_log_artifact,
)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    apply_mlflow_env_overrides(cfg)
    setup_mlflow(cfg)
    configure_run_repository(cfg.mlflow.tracking_uri, cfg.mlflow.experiment_name)

    if not cfg.analysis.consistency.enabled:
        print("Consistency computation disabled (analysis.consistency.enabled=false).")
        return

    split = cfg.execution.split
    force = cfg.execution.force
    anchors_dir = get_anchors_dir(cfg)
    anchors_cfg = cfg.profiling.anchors

    # 1. Get ground-truth labels
    labels = get_split_labels(cfg, split)

    anchors = get_or_create_anchors(
        labels=labels,
        source_split=anchors_cfg.source_split,
        per_class=anchors_cfg.per_class,
        strategy=anchors_cfg.strategy,
        order_by=anchors_cfg.order_by,
        num_classes=get_num_classes(cfg.dataset.name),
        artifacts_root=str(anchors_dir),
        dataset_name=cfg.dataset.name,
    )
    anchor_indices = anchors["anchor_indices"]
    a_spec_hash = anchors["spec_hash"]

    # 2. Discover ensemble groups dynamically from the actual DB tracking
    groups = discover_ensembles_from_cfg(cfg, cfg.mlflow.experiment_name)

    print(f"\nDiscovered {len(groups)} ensemble groups from MLflow.")

    for ens_name, run_ids in groups.items():
        print(f"\n{'='*60}")
        print(f"Consistency: {ens_name}")

        cs_hash = component_set_hash(run_ids)
        cons_hash = identity_hash(
            "consistency",
            component_set_hash=cs_hash,
            anchor_spec_hash=a_spec_hash,
            split=split,
        )

        # Idempotency
        if not force:
            existing = find_finished_identity_run("consistency", cons_hash)
            if existing is not None:
                print(f"  SKIP: already computed (run_id={existing.info.run_id})")
                continue

        print(f"  Components: {len(run_ids)} runs")

        # Compute per-model RDMs from anchor embeddings
        rdms = {}
        skip = False

        for run_id in run_ids:
            # Find the corresponding inference run
            inf_identity = identity_hash(
                "inference", trained_model_run_id=run_id, split=split
            )
            inf_run = find_finished_identity_run("inference", inf_identity)

            if inf_run is None:
                print(
                    f"  SKIP: embeddings not cached for {run_id} via related inference logs"
                )
                skip = True
                break

            inf_run_id = inf_run.info.run_id

            # 2. Download the tracked tensor artifact
            data = load_mlflow_artifact(
                inf_run_id,
                f"inference/{split}_tensors.npz",
                file_type="numpy",
                strict=True,
                cache_dir=cfg.mlflow.artifact_cache_dir,
            )
            embeddings = torch.from_numpy(data["embeddings"])

            anchor_embs = embeddings[anchor_indices]  # [K, D]
            rdm = pearson_rdm(anchor_embs)  # [K, K]
            rdms[run_id] = rdm

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

        # Log MLflow run
        tags = {
            "ensemble_name": ens_name,
            "component_set_hash": cs_hash,
            "consistency_hash": cons_hash,
            "identity_hash": cons_hash,
            "anchor_spec_hash": a_spec_hash,
            "run_name": f"cons_{ens_name}",
        }

        with schema_start_run(
            kind="consistency",
            run_name=f"cons_{ens_name}",
            tags=tags,
        ) as cons_run:
            schema_log_params(
                "consistency",
                {
                    "num_components": n_models,
                    "split": split,
                    "anchors_per_class": anchors_cfg.per_class,
                },
            )
            timed_log_metric("mean_rsa_correlation", mean_rsa)

            # Save and log all artifacts via tmpdir
            with tempfile.TemporaryDirectory() as tmpdir:
                for rid, rdm in rdms.items():
                    torch.save(rdm, os.path.join(tmpdir, f"{rid}_rdm.pt"))

                rsa_path = os.path.join(tmpdir, "rsa_matrix.pt")
                torch.save(torch.tensor(rsa_matrix, dtype=torch.float32), rsa_path)

                ordering_path = os.path.join(tmpdir, "run_id_ordering.json")
                with open(ordering_path, "w") as f:
                    json.dump(run_ids, f, indent=2)

                for fname in os.listdir(tmpdir):
                    fpath = os.path.join(tmpdir, fname)
                    if os.path.isfile(fpath):
                        timed_log_artifact(fpath, artifact_path="consistency")

            labels_subset = labels[anchor_indices]
            log_dataset_lineage(
                labels_subset,
                split,
                f"{cfg.dataset.name}_consistency_anchors",
                context="evaluation",
            )

        print(f"  Done. mean_rsa={mean_rsa:.4f}  run_id={cons_run.info.run_id}")

    print("\nConsistency computation complete.")


if __name__ == "__main__":
    main()
