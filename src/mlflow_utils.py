"""
MLflow helper utilities.

Provides:
  - ``cfg_hash``: re-exported from ``src.config.hash`` for convenience
  - ``setup_mlflow``: one-call setup from Hydra config
  - ``log_git_info``: logs git commit / dirty / diff as MLflow tags/artifacts
  - ``find_finished_run``: idempotency check
  - ``log_resolved_config``: log resolved Hydra config as artifact
  - Tag builder functions for model / behavior / manifest / profile runs
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Any, Dict, Optional

import mlflow
from omegaconf import DictConfig, OmegaConf

# ── Re-export cfg_hash from canonical location ──
from src.config.hash import cfg_hash  # noqa: F401
from src.config.paths import ensure_output_dirs


# ───────────────── setup ─────────────────


def setup_mlflow(cfg: DictConfig) -> None:
    """Configure MLflow tracking URI, experiment, and system metrics from Hydra config.
    
    Also ensures output directories exist before any MLflow operations.
    """
    # Ensure output directories exist (including parent of mlflow.db)
    ensure_output_dirs(cfg)
    
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    
    # Enable system metrics logging if configured (MLflow 2.8+)
    enable_system_metrics = getattr(cfg.mlflow, 'enable_system_metrics', False)
    if enable_system_metrics:
        try:
            mlflow.enable_system_metrics_logging()
        except AttributeError:
            # MLflow version doesn't support system metrics
            pass


def log_resolved_config(cfg: DictConfig) -> None:
    """Log the fully-resolved Hydra config as a YAML artifact."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
        f.flush()
        mlflow.log_artifact(f.name, artifact_path="config")
        os.unlink(f.name)


# ───────────────── git info ─────────────────


def log_git_info(run_dir: str = ".") -> None:
    """
    Log ``git_commit``, ``git_dirty`` tags, and optionally a ``git_diff.patch`` artifact.

    Best-effort: if git is unavailable or the directory is not a repo,
    tags are set to ``"unknown"`` / ``"unknown"`` and no artifact is logged.
    """
    try:
        import git

        repo = git.Repo(run_dir, search_parent_directories=True)
        sha = repo.head.commit.hexsha
        dirty = repo.is_dirty(untracked_files=True)
        mlflow.set_tag("git_commit", sha)
        mlflow.set_tag("git_dirty", str(dirty).lower())

        if dirty:
            diff_text = repo.git.diff()
            if diff_text:
                tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".patch", delete=False, prefix="git_diff_"
                )
                try:
                    tmp.write(diff_text)
                    tmp.close()
                    mlflow.log_artifact(tmp.name, artifact_path="git")
                finally:
                    os.unlink(tmp.name)
    except Exception:
        mlflow.set_tag("git_commit", "unknown")
        mlflow.set_tag("git_dirty", "unknown")


# ───────────────── idempotency ─────────────────


def find_finished_run(
    experiment_name: str,
    cfg_hash_value: str,
    kind: str | None = None,
) -> Optional[mlflow.entities.Run]:
    """
    Search for a FINISHED MLflow run matching ``cfg_hash``.

    Returns the run if found, else ``None``.
    """
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return None

    filter_parts = [
        f"tags.cfg_hash = '{cfg_hash_value}'",
        "attributes.status = 'FINISHED'",
    ]
    if kind:
        filter_parts.append(f"tags.kind = '{kind}'")
    filter_str = " and ".join(filter_parts)

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=filter_str,
        max_results=1,
        output_format="list",
    )
    return runs[0] if runs else None


# ───────────────── common tags ─────────────────


def _format_rho(rho) -> str:
    """Consistent string representation of rho for MLflow tags."""
    return str(float(rho))


def model_tags(cfg: DictConfig, cfg_hash_value: str, dataset_manifest_hash: str = "") -> Dict[str, str]:
    """Standard tag dict for a *model* training run."""
    tags = {
        "kind": "model",
        "schema_version": str(cfg.schema_version),
        "loss_type": "cross_entropy",
        "rho": _format_rho(cfg.loss.rho),
        "trial": str(cfg.trial),
        "seed": str(cfg.seed),
        "topography_type": str(cfg.loss.topography_type),
        "topology": str(cfg.loss.topology),
        "dataset": str(cfg.dataset.name),
        "model_arch": str(cfg.model.arch),
        "transforms_preset": str(cfg.dataset.transforms.preset),
        "cfg_hash": cfg_hash_value,
    }
    if dataset_manifest_hash:
        tags["dataset_manifest_hash"] = dataset_manifest_hash
    return tags


def behavior_tags(
    *,
    behavior: str,
    component_run_ids: list[str],
    behavior_input_hash: str,
    component_set_hash: str,
    extra: Dict[str, str] | None = None,
) -> Dict[str, str]:
    """Standard tag dict for a *behavior* (ensemble / meta-learner) run."""
    tags = {
        "kind": "behavior",
        "behavior": behavior,
        "component_set_hash": component_set_hash,
        "behavior_input_hash": behavior_input_hash,
        "num_components": str(len(component_run_ids)),
    }
    if extra:
        tags.update(extra)
    return tags


def dataset_manifest_tags(
    dataset_name: str,
    split: str,
    manifest_hash: str = "",
) -> Dict[str, str]:
    tags = {
        "kind": "dataset_manifest",
        "dataset": dataset_name,
        "split": split,
    }
    if manifest_hash:
        tags["dataset_manifest_hash"] = manifest_hash
    return tags


def profiles_tags(cfg_hash_value: str, profile_type: str) -> Dict[str, str]:
    return {
        "kind": "profiles",
        "cfg_hash": cfg_hash_value,
        "profile_type": profile_type,
    }


def component_set_hash(run_ids: list[str]) -> str:
    """Hash of sorted component model run_ids."""
    canonical = json.dumps(sorted(run_ids), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def behavior_input_hash(
    component_set_hash_val: str,
    dataset_manifest_hash: str = "",
    split: str = "test",
    feature_type: str = "logits",
    anchor_spec: str = "",
    meta_split_spec: str = "",
    similarity_metric: str = "",
    init_seed: str = "",
) -> str:
    """Derived hash summarising everything that changes behavior input data.

    The ``similarity_metric`` field is included so that switching from e.g.
    cosine to L2 produces a different hash (and therefore a new run).
    Default is ``""`` for backward-compatibility with logits-only runs.

    The ``init_seed`` field is included so that different adapter init seeds
    produce different hashes (and therefore separate runs).
    """
    parts = [
        component_set_hash_val,
        dataset_manifest_hash,
        split,
        feature_type,
        anchor_spec,
        meta_split_spec,
        similarity_metric,
        init_seed,
    ]
    canonical = json.dumps(parts, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ───────────────── per-step idempotency ─────────────────


def find_finished_inference_run(
    experiment_name: str,
    parent_run_id: str,
    split: str = "test",
) -> Optional[mlflow.entities.Run]:
    """Check if an inference run already exists for a parent model run."""
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return None
    filter_str = (
        f"tags.kind = 'inference' and "
        f"tags.parent_run_id = '{parent_run_id}' and "
        f"tags.split = '{split}' and "
        f"attributes.status = 'FINISHED'"
    )
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=filter_str,
        max_results=1,
        output_format="list",
    )
    return runs[0] if runs else None




def find_finished_behavior_run(
    experiment_name: str,
    behavior_input_hash_val: str,
    behavior: str = "",
) -> Optional[mlflow.entities.Run]:
    """Check if a behavior run (ensemble/adapter) already exists."""
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return None
    parts = [
        f"tags.kind = 'behavior'",
        f"tags.behavior_input_hash = '{behavior_input_hash_val}'",
        "attributes.status = 'FINISHED'",
    ]
    if behavior:
        parts.append(f"tags.behavior = '{behavior}'")
    filter_str = " and ".join(parts)
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=filter_str,
        max_results=1,
        output_format="list",
    )
    return runs[0] if runs else None


# ───────────────── category similarity profile ─────────────────


def category_similarity_profile_tags(
    parent_run_id: str,
    anchor_spec_hash: str,
    similarity_metric: str,
    split: str,
    profile_hash: str,
    extra: Dict[str, str] | None = None,
) -> Dict[str, str]:
    """Standard tag dict for a *category_similarity_profile* run."""
    tags = {
        "kind": "category_similarity_profile",
        "parent_run_id": parent_run_id,
        "anchor_spec_hash": anchor_spec_hash,
        "similarity_metric": similarity_metric,
        "split": split,
        "profile_hash": profile_hash,
    }
    if extra:
        tags.update(extra)
    return tags


def find_finished_similarity_profile_run(
    experiment_name: str,
    parent_run_id: str,
    anchor_spec_hash: str,
    similarity_metric: str,
    split: str = "test",
) -> Optional[mlflow.entities.Run]:
    """Check if a category_similarity_profile run already exists."""
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return None
    filter_str = (
        f"tags.kind = 'category_similarity_profile' and "
        f"tags.parent_run_id = '{parent_run_id}' and "
        f"tags.anchor_spec_hash = '{anchor_spec_hash}' and "
        f"tags.similarity_metric = '{similarity_metric}' and "
        f"tags.split = '{split}' and "
        f"attributes.status = 'FINISHED'"
    )
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=filter_str,
        max_results=1,
        output_format="list",
    )
    return runs[0] if runs else None
