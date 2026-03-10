#!/usr/bin/env python3
"""
03_compute_profiles.py — Category similarity profile computation.

For each FINISHED model run, compute per-sample category similarity profiles
against deterministic anchors.  Reads cached embeddings from step 02.

Each profiling job is tracked as its own MLflow run
(kind=category_similarity_profile), linked to the parent model run.

Uses manifest-driven anchor selection (canonical).

Usage:
    python scripts/03_compute_profiles.py
    python scripts/03_compute_profiles.py pipeline.force=true
    python scripts/03_compute_profiles.py pipeline.anchors.per_class=200
"""

from __future__ import annotations

import os

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.anchors import get_or_create_anchors
from src.data.cache import get_backend
from src.data.manifest import get_or_create_manifest
from src.ensemble.selector import resolve_components
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
    """Extract unique (anchor_selection, similarity_metric) combos from all meta defs.

    Returns a list of dicts with keys:
      - anchor_selection: dict
      - similarity_metric: str
    """
    default_sel = OmegaConf.to_container(cfg.ensemble.default_anchor_selection, resolve=True)
    ensembles = OmegaConf.to_container(cfg.ensemble.ensembles, resolve=True)

    seen = set()
    specs = []

    for ens_def in ensembles:
        for meta_def in ens_def.get("meta", []):
            ft = meta_def.get("feature_type", "logits")
            if ft not in ("embeddings", "embeddings+profiles"):
                continue  # logits-only → no profiles needed

            sim = meta_def.get("similarity_metric", "cosine")
            anchor_sel = meta_def.get("anchor_selection") or default_sel

            # Deduplicate by (anchor_sel canonical, metric)
            key = (
                tuple(sorted(anchor_sel.items())),
                sim,
            )
            if key not in seen:
                seen.add(key)
                specs.append({
                    "anchor_selection": anchor_sel,
                    "similarity_metric": sim,
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
    """Compute profiles for one (anchor_selection, metric) spec across all model runs.

    Returns (computed_count, skipped_count).
    """
    anchor_sel = spec["anchor_selection"]
    similarity_metric = spec["similarity_metric"]

    anchors = get_or_create_anchors(
        manifest,
        per_class=anchor_sel.get("per_class", 100),
        strategy=anchor_sel.get("strategy", "per_class_first_n"),
        order_by=anchor_sel.get("order_by", "example_id"),
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
    artifacts_root = cfg.runtime.artifacts_root

    # Manifest
    manifest = get_or_create_manifest(
        dataset_name=cfg.dataset.name,
        split=split,
        data_root=cfg.runtime.data_root,
        artifacts_root=artifacts_root,
    )

    # Collect all unique (anchor_selection, similarity_metric) specs
    specs = _collect_profile_specs(cfg)
    if not specs:
        print("No meta-learner definitions require similarity profiles. Nothing to do.")
        return

    print(f"Found {len(specs)} unique (anchor_selection, metric) combinations to compute.")
    for i, s in enumerate(specs):
        print(f"  [{i}] metric={s['similarity_metric']}  anchors={s['anchor_selection']}")

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
        print(f"Computing: metric={spec['similarity_metric']}  anchors={spec['anchor_selection']}")
        computed, skipped = _compute_for_spec(
            cfg, spec, model_runs, manifest, artifacts_root, split, force
        )
        total_computed += computed
        total_skipped += skipped

    print(f"\nProfile computation complete. Computed: {total_computed}  Skipped: {total_skipped}")


if __name__ == "__main__":
    main()
