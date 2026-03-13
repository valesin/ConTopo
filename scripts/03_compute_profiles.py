#!/usr/bin/env python3
"""
03_compute_profiles.py — Category similarity profile computation.

For each FINISHED model run, compute per-sample category similarity profiles
against deterministic anchors.  Reads cached embeddings from step 02.

Each profiling job is tracked as its own MLflow run
(kind=category_similarity_profile), linked to the parent model run.

Anchor selection and similarity metric are read from ``cfg.pipeline``:
  - ``pipeline.anchors``          (per_class, strategy, order_by, source_split)
  - ``pipeline.profiles.metrics`` (list of metrics, e.g. [cosine, l2])

Profiles are **always computed** for all FINISHED model runs, regardless of
the current ``adapter.feature_type`` setting.  This is intentional:
downstream steps (e.g. Step 5 adapter training) may run with a different
``feature_type`` that requires pre-computed profiles.  Pipeline steps that
generate cross-config reusable artifacts must not gate on the current config.

To explicitly skip profile computation, set ``pipeline.profiles.skip=true``.

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
import torch
from omegaconf import DictConfig

from src.data.anchors import AnchorSpec, get_or_create_anchors
from src.data.cache import get_backend
from src.data.manifest import get_or_create_manifest
from src.config.paths import get_cache_dir
from src.profiling.category_similarity import (
    compute_similarity_profile,
    similarity_profile_hash,
)
from src.mlflow_utils import (
    category_similarity_profile_tags,
    find_finished_similarity_profile_run,
    log_git_info,
    log_resolved_config,
    setup_mlflow,
)


def _collect_profile_specs(cfg: DictConfig) -> list[dict]:
    """Build the list of profile specs from ``cfg.pipeline``.

    Always returns a list of spec dicts so that profiles are pre-computed
    for all model runs regardless of the current ``adapter.feature_type``.
    Downstream steps (e.g. Step 5 adapter training) may later run with
    ``feature_type=embeddings+profiles`` even when the current config uses
    ``logits`` or ``embeddings``.

    Pipeline best practice: cross-config reusable artifacts must not be
    gated on the current run configuration.
    """
    sel = cfg.pipeline.anchors
    anchor_spec = AnchorSpec(
        source_split=sel.source_split,
        per_class=sel.per_class,
        strategy=sel.strategy,
        order_by=sel.order_by,
        num_classes=cfg.dataset.num_classes,
    )
    
    specs = []
    for metric in cfg.pipeline.profiles.metrics:
        specs.append({
            "anchor_spec": anchor_spec,
            "similarity_metric": metric,
        })

    return specs


def _compute_for_spec(
    cfg: DictConfig,
    spec: dict,
    model_runs: list,
    manifest,
    artifacts_root: str,
    split: str,
    force: bool,
) -> tuple[int, int]:
    """Compute profiles for one (anchor_spec, metric) spec across all model runs.

    Returns (computed_count, skipped_count).
    """
    anchor_spec = spec["anchor_spec"]
    similarity_metric = spec["similarity_metric"]

    anchors = get_or_create_anchors(
        manifest,
        spec=anchor_spec,
        artifacts_root=artifacts_root,
    )
    a_spec_hash = anchors["spec_hash"]
    anchor_indices = anchors["anchor_indices"]

    backend = get_backend("pt")
    computed = 0
    skipped = 0

    for run in model_runs:
        run_id = run.info.run_id
        prof_hash = similarity_profile_hash(run_id, a_spec_hash, similarity_metric, split)

        # Idempotency check
        if not force:
            existing = find_finished_similarity_profile_run(
                cfg.mlflow.experiment_name, run_id, a_spec_hash, similarity_metric, split
            )
            if existing is not None:
                skipped += 1
                continue

        # Check local cache
        profile_dir = os.path.join(
            artifacts_root, "similarity_profiles", run_id, a_spec_hash, similarity_metric, split
        )
        profile_path = os.path.join(profile_dir, "profiles.pt")

        if not force and os.path.isfile(profile_path):
            # Local cache exists but no MLflow run — register it
            profiles = torch.load(profile_path, weights_only=True)
        else:
            # Compute from scratch
            emb_path = os.path.join(
                artifacts_root, "inference", run_id, split, f"embeddings{backend.extension}"
            )
            if not backend.exists(emb_path):
                rho = run.data.tags.get("rho", "?")
                trial = run.data.tags.get("trial", "?")
                print(f"  SKIP rho={rho} trial={trial}: embeddings not cached.")
                skipped += 1
                continue

            embeddings = backend.load(emb_path)
            anchor_embeddings = embeddings[anchor_indices]
            profiles = compute_similarity_profile(
                embeddings, anchor_embeddings, metric=similarity_metric
            )

            # Save locally
            os.makedirs(profile_dir, exist_ok=True)
            torch.save(profiles, profile_path)

        # Log as MLflow run
        rho = run.data.tags.get("rho", "?")
        trial = run.data.tags.get("trial", "?")
        topology = run.data.tags.get("topology", "?")

        tags = category_similarity_profile_tags(
            parent_run_id=run_id,
            anchor_spec_hash=a_spec_hash,
            similarity_metric=similarity_metric,
            split=split,
            profile_hash=prof_hash,
        )

        with mlflow.start_run(
            run_name=f"csp_{similarity_metric}_{topology}_rho{rho}_t{trial}",
            tags=tags,
        ):
            mlflow.log_params({
                "parent_run_id": run_id,
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
            })
            mlflow.log_artifact(profile_path, artifact_path="profiles")
            log_git_info()

        computed += 1
        print(f"  Computed profile rho={rho} trial={trial} metric={similarity_metric}")

    return computed, skipped


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_mlflow(cfg)

    split = cfg.pipeline.split
    force = cfg.pipeline.force
    cache_dir = get_cache_dir(cfg)

    # Manifest
    manifest = get_or_create_manifest(
        dataset_name=cfg.dataset.name,
        split=split,
        data_root=cfg.runtime.data_root,
        artifacts_root=str(cache_dir),
    )

    # Explicit user-intent skip flag (default: false, via structured config)
    if cfg.pipeline.profiles.skip:
        print("Profile computation skipped (pipeline.profiles.skip=true).")
        return

    # Collect all unique (anchor_spec, similarity_metric) specs
    # Always generates profiles — not gated on adapter.feature_type
    specs = _collect_profile_specs(cfg)

    print(f"Found {len(specs)} unique (anchor_spec, metric) combinations to compute.")
    for i, s in enumerate(specs):
        print(f"  [{i}] metric={s['similarity_metric']}  anchors={s['anchor_spec']}")

    # Get all FINISHED model runs
    exp = mlflow.get_experiment_by_name(cfg.mlflow.experiment_name)
    if exp is None:
        print(f"Experiment '{cfg.mlflow.experiment_name}' not found.")
        return

    model_runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.kind = 'model' and attributes.status = 'FINISHED'",
        output_format="list",
    )
    print(f"Found {len(model_runs)} model runs.")

    total_computed = 0
    total_skipped = 0

    for spec in specs:
        print(f"\n{'─'*50}")
        print(f"Computing: metric={spec['similarity_metric']}  anchors={spec['anchor_spec']}")
        computed, skipped = _compute_for_spec(
            cfg, spec, model_runs, manifest, str(cache_dir), split, force
        )
        total_computed += computed
        total_skipped += skipped

    print(f"\nProfile computation complete. Computed: {total_computed}  Skipped: {total_skipped}")


if __name__ == "__main__":
    main()
