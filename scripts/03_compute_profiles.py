#!/usr/bin/env python3
"""
03_compute_profiles.py — Category similarity profile computation.

For the FINISHED model run matching the current configuration, compute per-sample
category similarity profiles against deterministic anchors. Reads cached embeddings
from step 02 via MLflow artifacts.

Each profiling job is tracked as its own MLflow run
(kind=category_similarity_profile), linked to the parent model run and inference run.

Anchor selection and similarity metric are read from `cfg.pipeline`:
  - `pipeline.anchors`          (per_class, strategy, order_by, source_split)
  - `pipeline.profiles.metrics` (list of metrics, e.g. [cosine, l2])

Usage:
    python scripts/03_compute_profiles.py
    python scripts/03_compute_profiles.py pipeline.force=true
    python scripts/03_compute_profiles.py pipeline.anchors.per_class=200
    python scripts/03_compute_profiles.py "pipeline.profiles.metrics=[l2]"
    python scripts/03_compute_profiles.py pipeline.profiles.skip=true
"""

from __future__ import annotations
import os
import hydra
import mlflow
import mlflow.artifacts
import torch
from omegaconf import DictConfig

from src.data.anchors import AnchorSpec, get_or_create_anchors
from src.data.manifest import get_or_create_manifest
from src.config.paths import get_cache_dir
from src.profiling.category_similarity import (
    compute_similarity_profile,
    similarity_profile_hash,
)
from src.mlflow_utils import (
    category_similarity_profile_tags,
    find_finished_similarity_profile_run,
    log_resolved_config,
    setup_mlflow,
    get_existing_model,
    resolve_seed,
    get_inference_run,
    load_mlflow_artifact,
    log_manifest_lineage,
)
from src.config.hash import cfg_hash


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_mlflow(cfg)

    # ── Seed ──
    seed = resolve_seed(cfg)
    cfg.seed = seed

    if cfg.pipeline.profiles.skip:
        print("Profile computation skipped (pipeline.profiles.skip=true).")
        return

    # ── Target specific model by ID ──
    hash_val = cfg_hash(cfg)
    model, run_id = get_existing_model(cfg.mlflow.experiment_name, hash_val)

    if run_id is None:
        print(
            f"A model with cfg_hash={hash_val} has not been trained yet. Please run 01_train_models.py first."
        )
        return

    split = cfg.pipeline.split
    force = cfg.pipeline.force
    cache_dir = get_cache_dir(cfg)

    # ── Fetch corresponding Inference Run ──
    # ── Fetch corresponding Inference Run ──
    inf_runs = get_inference_run(cfg.mlflow.experiment_name, run_id, split)

    if len(inf_runs) == 0:
        print(
            f"No inference run found for model {run_id} on split '{split}'. Please run 02_cache_inference.py first."
        )
        return

    inf_run_id = inf_runs.iloc[0].run_id

    # ── Manifest & Anchors ──
    manifest = get_or_create_manifest(
        dataset_name=cfg.dataset.name,
        split=split,
        data_root=cfg.runtime.data_root,
        artifacts_root=str(cache_dir),
    )

    sel = cfg.pipeline.anchors
    anchor_spec = AnchorSpec(
        source_split=sel.source_split,
        per_class=sel.per_class,
        strategy=sel.strategy,
        order_by=sel.order_by,
        num_classes=cfg.dataset.num_classes,
    )

    anchors = get_or_create_anchors(
        manifest,
        spec=anchor_spec,
        artifacts_root=str(cache_dir),
    )
    a_spec_hash = anchors["spec_hash"]
    anchor_indices = anchors["anchor_indices"]

    print(
        f"Anchors identified. Total anchors: {len(anchor_indices)}. Spec Hash: {a_spec_hash}"
    )

    # ── Load Inference Embeddings from MLflow Artifact ──
    try:
        data = load_mlflow_artifact(inf_run_id, f"inference_data/{split}_tensors.npz", file_type="numpy")
        embeddings = torch.from_numpy(data["embeddings"])
    except Exception as e:
        print(
            f"Failed to load embeddings artifact from tracking db for run {inf_run_id}: {e}"
        )
        return

    anchor_embeddings = embeddings[anchor_indices]

    # Get parent model tags for logging
    model_run = mlflow.get_run(run_id)
    rho = model_run.data.tags.get("rho", "?")
    trial = model_run.data.tags.get("trial", "?")
    topology = model_run.data.tags.get("topology", "?")

    computed = 0
    skipped = 0

    # Should this be changed and become only for one metric and sweepable?
    # ── Compute and Log Profiles for Each Metric ──
    for similarity_metric in cfg.pipeline.profiles.metrics:
        print(f"\nProcessing metric={similarity_metric}")
        prof_hash = similarity_profile_hash(
            run_id, a_spec_hash, similarity_metric, split
        )

        # Idempotency check check across MLflow backend
        if not force:
            existing = find_finished_similarity_profile_run(
                cfg.mlflow.experiment_name,
                run_id,
                a_spec_hash,
                similarity_metric,
                split,
            )
            if existing is not None:
                print(
                    f"  -> Profile {similarity_metric} already computed for model {run_id}. Skipping."
                )
                skipped += 1
                continue

        print(f"  -> Computing profiles for metric '{similarity_metric}'...")
        profiles = compute_similarity_profile(
            embeddings, anchor_embeddings, num_classes=cfg.dataset.num_classes, metric=similarity_metric
        )

        tags = category_similarity_profile_tags(
            parent_run_id=run_id,
            anchor_spec_hash=a_spec_hash,
            similarity_metric=similarity_metric,
            split=split,
            profile_hash=prof_hash,
        )
        tags["inference_run_id"] = inf_run_id

        # Profile run instance logging
        run_name = f"csp_{similarity_metric}_{topology}_rho{rho}_t{trial}"
        with mlflow.start_run(run_name=run_name, tags=tags):
            mlflow.log_params(
                {
                    "parent_run_id": run_id,
                    "inference_run_id": inf_run_id,
                    "anchor_spec_hash": a_spec_hash,
                    "similarity_metric": similarity_metric,
                    "split": split,
                    "profile_hash": prof_hash,
                    "num_anchors": len(anchor_indices),
                    "num_samples": int(profiles.shape[0]),
                    "profile_dim": int(profiles.shape[1]),
                    "rho": rho,
                    "trial": trial,
                    "topology": topology,
                }
            )

            # Save strictly inside the global cache, exactly like 02_cache_inference
            os.makedirs(cache_dir, exist_ok=True)
            profile_path = os.path.join(
                cache_dir, f"{split}_{similarity_metric}_profiles.pt"
            )
            torch.save(profiles, profile_path)

            # ── Log the dataset manifest to MLflow for lineage ──
            log_manifest_lineage(manifest, split, cfg.dataset.name, context="profiling")

            # Upload as MLflow Artifact Tracking
            mlflow.log_artifact(profile_path, artifact_path="profiles")

            log_resolved_config(cfg)

        computed += 1

    print(f"\nDone. Computed: {computed} | Skipped: {skipped}")


if __name__ == "__main__":
    main()
