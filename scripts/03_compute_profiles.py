#!/usr/bin/env python3
"""
03_compute_profiles.py — Category similarity profile computation.

For the FINISHED model run matching the current configuration, compute per-sample
category similarity profiles against deterministic anchors. Reads cached embeddings
from step 02 via MLflow artifacts.

Each profiling job is tracked as its own MLflow run
(kind=category_similarity_profile), linked to the parent model run and inference run.

Anchor selection and similarity metric are read from `cfg.profiling`:
  - `profiling.anchors`          (per_class, strategy, order_by, source_split)
  - `profiling.profiles.metrics` (list of metrics, e.g. [cosine, l2])

Usage:
    python scripts/03_compute_profiles.py
    python scripts/03_compute_profiles.py execution.force=true
    python scripts/03_compute_profiles.py profiling.anchors.per_class=200
    python scripts/03_compute_profiles.py "profiling.profiles.metrics=[l2]"
    python scripts/03_compute_profiles.py profiling.profiles.skip=true
"""

from __future__ import annotations
import os
import tempfile
import hydra
import mlflow
import torch
from omegaconf import DictConfig

from src.data.anchors import get_or_create_anchors
from src.data.loaders import get_num_classes, get_split_labels
from src.config.paths import get_anchors_dir
from src.profiling.category_similarity import (
    compute_similarity_profile,
)
from src.mlflow_utils import (
    apply_mlflow_env_overrides,
    category_similarity_profile_tags,
    log_resolved_config,
    setup_mlflow,
    resolve_seed,
    load_mlflow_artifact,
    get_run_context,
    log_dataset_lineage,
)
from src.repositories.functional_run_repository import (
    configure_run_repository,
    find_finished_model_run,
    find_finished_identity_run,
)
from src.config.hash import identity_hash
from src.mlflow_schema_logger import (
    log_params as schema_log_params,
    start_run as schema_start_run,
    timed_log_artifact,
)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    apply_mlflow_env_overrides(cfg)
    setup_mlflow(cfg)
    configure_run_repository(cfg.mlflow.tracking_uri, cfg.mlflow.experiment_name)

    # ── Seed ──
    seed = resolve_seed(cfg)
    cfg.seed = seed

    if cfg.profiling.profiles.skip:
        print("Profile computation skipped (profiling.profiles.skip=true).")
        return

    # ── Target specific model by ID ──
    model_run, _ = find_finished_model_run(cfg, seed)

    if model_run is None:
        print(
            "No trained model found for this config. Please run 01_train_models.py first."
        )
        return
    run_id = model_run.info.run_id

    split = cfg.execution.split
    force = cfg.execution.force
    anchors_dir = get_anchors_dir(cfg)

    # ── Fetch corresponding Inference Run ──
    inf_identity = identity_hash("inference", trained_model_run_id=run_id, split=split)
    inf_run = find_finished_identity_run("inference", inf_identity)

    if inf_run is None:
        print(
            f"No inference run found for model {run_id} on split '{split}'. Please run 02_cache_inference.py first."
        )
        return

    inf_run_id = inf_run.info.run_id

    # ── Labels & Anchors ──
    labels = get_split_labels(cfg, split)

    sel = cfg.profiling.anchors
    anchors = get_or_create_anchors(
        labels=labels,
        source_split=sel.source_split,
        per_class=sel.per_class,
        strategy=sel.strategy,
        order_by=sel.order_by,
        num_classes=get_num_classes(cfg.dataset.name),
        artifacts_root=str(anchors_dir),
        dataset_name=cfg.dataset.name,
    )
    a_spec_hash = anchors["spec_hash"]
    anchor_indices = anchors["anchor_indices"]

    print(
        f"Anchors identified. Total anchors: {len(anchor_indices)}. Spec Hash: {a_spec_hash}"
    )

    # ── Load Inference Embeddings from MLflow Artifact ──
    try:
        data = load_mlflow_artifact(
            inf_run_id,
            f"inference/{split}_tensors.npz",
            file_type="numpy",
            strict=True,
            cache_dir=cfg.mlflow.artifact_cache_dir,
        )
        embeddings = torch.from_numpy(data["embeddings"])
    except Exception as e:
        print(
            f"Failed to load embeddings artifact from tracking db for run {inf_run_id}: {e}"
        )
        return

    # Anchor embeddings are sliced from the same inference run as the profiled samples,
    # so source_split must equal execution.split. Using train-split anchors (to avoid
    # anchors appearing among the profiled samples) is not implemented: it would require
    # a separate inference run for the train split, loading its embeddings independently,
    # and using those as anchor_embeddings instead.
    anchor_embeddings = embeddings[anchor_indices]

    # Get parent model tags for logging
    model_run = mlflow.get_run(run_id)
    rho, trial, topology = get_run_context(model_run)

    computed = 0
    skipped = 0

    # Should this be changed and become only for one metric and sweepable?
    # ── Compute and Log Profiles for Each Metric ──
    for similarity_metric in cfg.profiling.profiles.metrics:
        print(f"\nProcessing metric={similarity_metric}")
        prof_hash = identity_hash(
            "category_similarity_profile",
            parent_run_id=run_id,
            anchor_spec_hash=a_spec_hash,
            similarity_metric=similarity_metric,
            split=split,
        )

        # Idempotency check across MLflow backend
        if not force:
            existing = find_finished_identity_run(
                "category_similarity_profile",
                prof_hash,
            )
            if existing is not None:
                print(
                    f"  -> Profile {similarity_metric} already computed for model {run_id}. Skipping."
                )
                skipped += 1
                continue

        print(f"  -> Computing profiles for metric '{similarity_metric}'...")
        profiles = compute_similarity_profile(
            embeddings,
            anchor_embeddings,
            num_classes=get_num_classes(cfg.dataset.name),
            metric=similarity_metric,
        )

        tags = category_similarity_profile_tags(
            parent_run_id=run_id,
            anchor_spec_hash=a_spec_hash,
            identity_hash=prof_hash,
            similarity_metric=similarity_metric,
            split=split,
            profile_hash=prof_hash,
        )
        run_name = f"csp_{similarity_metric}_{topology}_rho{rho}_t{trial}"
        tags["inference_run_id"] = inf_run_id
        tags["profile_dim"] = int(profiles.shape[1])
        tags["run_name"] = run_name

        # Profile run instance logging
        with schema_start_run(
            kind="category_similarity_profile", run_name=run_name, tags=tags
        ):
            schema_log_params(
                "category_similarity_profile",
                {
                    "similarity_metric": similarity_metric,
                    "split": split,
                    "profile_hash": prof_hash,
                    "num_anchors": len(anchor_indices),
                    "num_samples": int(profiles.shape[0]),
                    "rho": rho,
                    "trial": trial,
                    "topology": topology,
                },
            )

            # Save to tmpdir and upload as MLflow artifact
            with tempfile.TemporaryDirectory() as tmpdir:
                profile_path = os.path.join(
                    tmpdir, f"{split}_{similarity_metric}_profiles.pt"
                )
                torch.save(profiles, profile_path)
                timed_log_artifact(profile_path, artifact_path="profiles")

            log_dataset_lineage(labels, split, cfg.dataset.name, context="profiling")

            log_resolved_config(cfg)

        computed += 1

    print(f"\nDone. Computed: {computed} | Skipped: {skipped}")


if __name__ == "__main__":
    main()
