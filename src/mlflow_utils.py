"""
MLflow helper utilities.

Provides:
  - ``cfg_hash``: re-exported from ``src.config.hash`` for convenience
  - ``setup_mlflow``: one-call setup from Hydra config
  - ``log_git_info``: logs git commit / dirty / diff as MLflow tags/artifacts
  - ``find_finished_run``: idempotency check
  - ``log_resolved_config``: log resolved Hydra config as artifact
  - Tag builder functions for model / behaviour / profile runs
"""

from __future__ import annotations

import os
import logging
import tempfile
import pandas as pd
from datetime import datetime
from typing import Any, Dict, Optional

import mlflow
from mlflow.tracking import MlflowClient
import torch
import numpy as np

from omegaconf import DictConfig, OmegaConf

# ── Re-export cfg_hash from canonical location ──
from src.config.hash import (  # noqa: F401
    cfg_hash,
    component_set_hash as _component_set_hash,
    identity_hash as _identity_hash,
    model_identity_fields as _model_identity_fields,
)
from src.config.paths import ensure_output_dirs


def _resolve_artifact_cache_dir(cache_dir: str | None = None) -> str:
    if not cache_dir:
        raise ValueError(
            "cache_dir must be provided when use_cache=True. "
            "Pass cfg.mlflow.artifact_cache_dir from Hydra config."
        )
    resolved = os.path.abspath(os.path.expanduser(str(cache_dir)))
    os.makedirs(resolved, exist_ok=True)
    return resolved


def _sanitize_artifact_path(artifact_path: str) -> str:
    if not artifact_path:
        raise ValueError("artifact_path must be non-empty")

    normalized = os.path.normpath(artifact_path).replace("\\", "/")
    if normalized.startswith("../") or normalized == ".." or os.path.isabs(normalized):
        raise ValueError(f"Unsafe artifact_path: {artifact_path}")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


# ── Backward-compatibility: artifact path rename map ──────────────────────
# In April 2025 the pipeline artifact paths were normalised:
#   inference_data/ → inference/   (matches run kind)
#   ensemble_data/  → ensemble/    (matches run kind)
#   adapter_inputs/ → inputs/      (drop redundant prefix inside metalearner run)
#   adapter_data/   → data/        (drop redundant prefix inside metalearner run)
#
# Runs executed *before* that change still store artifacts under the old names.
# _legacy_artifact_path() maps a new-style path to its old equivalent so that
# load_mlflow_artifact() can transparently fall back when the primary URI is absent.
_LEGACY_ARTIFACT_PATHS: dict[str, str] = {
    "inference/": "inference_data/",
    "ensemble/": "ensemble_data/",
    "inputs/": "adapter_inputs/",
    "data/": "adapter_data/",
}


def _legacy_artifact_path(artifact_path: str) -> str | None:
    """Return the pre-rename equivalent of *artifact_path*, or None if no mapping exists."""
    for new_prefix, old_prefix in _LEGACY_ARTIFACT_PATHS.items():
        if artifact_path.startswith(new_prefix):
            return old_prefix + artifact_path[len(new_prefix) :]
    return None


def _download_artifact_uri(artifact_uri: str, dst_path: str | None = None) -> str:
    """Download an artifact by URI, falling back to the pre-rename path on MlflowException.

    This is the single choke-point for all artifact downloads inside
    load_mlflow_artifact().  Callers never need to know about legacy paths.
    """
    from mlflow.exceptions import MlflowException

    kwargs: dict = {"artifact_uri": artifact_uri}
    if dst_path is not None:
        kwargs["dst_path"] = dst_path

    try:
        return mlflow.artifacts.download_artifacts(**kwargs)
    except MlflowException:
        # Extract the artifact_path portion from "runs:/{run_id}/{artifact_path}"
        prefix = "runs:/"
        if not artifact_uri.startswith(prefix):
            raise  # Not a run-relative URI — nothing we can do

        run_id_part, _, artifact_path_part = artifact_uri[len(prefix) :].partition("/")
        legacy_path = _legacy_artifact_path(artifact_path_part)
        if legacy_path is None:
            raise  # No mapping for this path — re-raise the original error

        legacy_uri = f"runs:/{run_id_part}/{legacy_path}"
        logging.getLogger(__name__).warning(
            "Artifact not found at '%s'; retrying with legacy path '%s'. "
            "Consider re-running the producing script to migrate this run.",
            artifact_uri,
            legacy_uri,
        )
        kwargs["artifact_uri"] = legacy_uri
        return mlflow.artifacts.download_artifacts(**kwargs)


def _load_artifact_from_local_path(
    local_path: str,
    run_id: str,
    artifact_path: str,
    file_type: str,
    strict: bool,
) -> Any:
    if file_type == "numpy":
        return np.load(local_path)
    if file_type == "torch":
        try:
            return torch.load(local_path, weights_only=False)
        except Exception as e:
            if strict:
                raise RuntimeError(
                    f"Failed to load torch artifact {artifact_path} for run {run_id}. {e}"
                )
            raise
    if file_type == "parquet":
        try:
            return pd.read_parquet(local_path)
        except Exception as e:
            if strict:
                raise RuntimeError(
                    f"Failed to load parquet artifact {artifact_path} for run {run_id}. {e}"
                )
            raise
    raise ValueError(f"Unsupported file_type: {file_type}")


# ───────────────── setup ─────────────────


def setup_mlflow(cfg: DictConfig) -> None:
    """Configure MLflow tracking URI, experiment, and system metrics from Hydra config.

    Also ensures output directories exist before any MLflow operations.
    """
    # Ensure output directories exist
    ensure_output_dirs(cfg)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    experiment_name = cfg.mlflow.experiment_name

    # Check if experiment exists, if not, create it with a custom artifact location
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        # Build an absolute URI for the configured or default artifact path
        artifact_loc = cfg.mlflow.get("artifact_location", "outputs/mlruns")
        if not (artifact_loc.startswith("file://") or "://" in artifact_loc):
            artifact_loc = f"file://{os.path.abspath(artifact_loc)}"

        if artifact_loc.startswith("file://"):
            os.makedirs(artifact_loc.replace("file://", "", 1), exist_ok=True)

        mlflow.create_experiment(name=experiment_name, artifact_location=artifact_loc)

    mlflow.set_experiment(experiment_name)

    # Enable system metrics logging if configured (MLflow 2.8+)
    enable_system_metrics = getattr(cfg.mlflow, "enable_system_metrics", False)
    if enable_system_metrics:
        try:
            mlflow.enable_system_metrics_logging()
        except AttributeError:
            # MLflow version doesn't support system metrics
            pass

    # ── Suppress Noisy Loggers ──
    # Hide the PyTorch pickling warning and environment resolution noise
    logging.getLogger("mlflow.pytorch").setLevel(logging.ERROR)
    logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)

    # Hide "Started monitoring system metrics" INFO logs
    logging.getLogger("mlflow.system_metrics").setLevel(logging.WARNING)


def log_resolved_config(cfg: DictConfig) -> None:
    """Log the fully-resolved Hydra config as a YAML artifact."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
        f.flush()
        start = datetime.now()
        print(
            f"[UPLOAD START] {start.strftime('%H:%M:%S')} — Logging artifact: config/resolved_config.yaml"
        )
        mlflow.log_artifact(f.name, artifact_path="config")
        end = datetime.now()
        elapsed = (end - start).total_seconds()
        print(
            f"[UPLOAD  END ] {end.strftime('%H:%M:%S')} — Logging artifact: config/resolved_config.yaml completed in {elapsed:.1f}s"
        )
        os.unlink(f.name)


def load_mlflow_artifact(
    run_id: str,
    artifact_path: str,
    file_type: str = "auto",
    strict: bool = True,
    *,
    cache_dir: str | None = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    validate_cache: bool = True,
) -> Any:
    """
    Download and load a specific artifact from an MLflow run.
    Supported types: 'auto', 'numpy', 'torch', 'parquet'.
    If strict=True, raises RuntimeError natively on failure to simplify caller scripts.
    """
    artifact_uri = f"runs:/{run_id}/{artifact_path}"

    cache_root = None
    normalized_artifact_path = _sanitize_artifact_path(artifact_path)

    if use_cache:
        cache_root = _resolve_artifact_cache_dir(cache_dir)
        local_path = os.path.join(cache_root, run_id, normalized_artifact_path)

        if (not refresh_cache) and os.path.exists(local_path):
            pass
        else:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            downloaded = _download_artifact_uri(
                artifact_uri, dst_path=os.path.dirname(local_path)
            )
            if os.path.exists(local_path):
                pass
            else:
                local_path = downloaded
    else:
        local_path = _download_artifact_uri(artifact_uri)

    if file_type == "auto":
        if normalized_artifact_path.endswith(
            ".npz"
        ) or normalized_artifact_path.endswith(".npy"):
            file_type = "numpy"
        elif normalized_artifact_path.endswith(
            ".pt"
        ) or normalized_artifact_path.endswith(".pth"):
            file_type = "torch"
        elif normalized_artifact_path.endswith(".parquet"):
            file_type = "parquet"
        else:
            return local_path  # Return the downloaded path if type is unknown

    try:
        return _load_artifact_from_local_path(
            local_path=local_path,
            run_id=run_id,
            artifact_path=artifact_path,
            file_type=file_type,
            strict=strict,
        )
    except Exception:
        if not (use_cache and validate_cache and cache_root is not None):
            raise

        # Retry once after cache refresh; intended for corrupted local cache files.
        if os.path.exists(local_path):
            try:
                if os.path.isdir(local_path):
                    import shutil

                    shutil.rmtree(local_path)
                else:
                    os.unlink(local_path)
            except OSError:
                pass
        local_path = os.path.join(cache_root, run_id, normalized_artifact_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        downloaded = _download_artifact_uri(
            artifact_uri, dst_path=os.path.dirname(local_path)
        )
        if not os.path.exists(local_path):
            local_path = downloaded

        return _load_artifact_from_local_path(
            local_path=local_path,
            run_id=run_id,
            artifact_path=artifact_path,
            file_type=file_type,
            strict=strict,
        )


def safe_to_numpy_float64(tensor_or_numpy):
    """
    Helper function to safely convert a PyTorch tensor (or already numpy array)
    to a numpy float64 array, standardizing inputs to MLflow data schema tracing.
    """
    arr = (
        tensor_or_numpy.numpy()
        if hasattr(tensor_or_numpy, "numpy")
        else tensor_or_numpy
    )
    return arr.astype("float64")


def log_dataset_lineage(
    labels: "torch.Tensor", split: str, dataset_name: str, context: str = "evaluation"
) -> None:
    """Log the corresponding dataset split as an MLflow input dataset (without image hashes)."""
    import pandas as pd

    dataset_df = pd.DataFrame(
        {
            "original_index": safe_to_numpy_float64(torch.arange(len(labels))),
            "label": safe_to_numpy_float64(labels),
        }
    )
    eval_dataset = mlflow.data.from_pandas(
        dataset_df, targets="label", name=f"{dataset_name}_{split}"
    )
    start = datetime.now()
    print(
        f"[UPLOAD START] {start.strftime('%H:%M:%S')} — Logging dataset lineage: {dataset_name}_{split} (context={context})"
    )
    mlflow.log_input(eval_dataset, context=context)
    end = datetime.now()
    elapsed = (end - start).total_seconds()
    print(
        f"[UPLOAD  END ] {end.strftime('%H:%M:%S')} — Logging dataset lineage: {dataset_name}_{split} (context={context}) completed in {elapsed:.1f}s"
    )


# ───────────────── idempotency ─────────────────


def resolve_seed(cfg: DictConfig) -> int:
    if cfg.seed is not None:
        return int(cfg.seed)
    return 100 + int(cfg.trial)


def set_torch_seed(seed: int) -> None:
    """Set torch CPU/CUDA seeds consistently."""
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(device_name: str) -> torch.device:
    """Resolve runtime device name into torch.device."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def get_run_context(run: mlflow.entities.Run) -> tuple[str, str, str]:
    """Extract common run metadata as (rho, trial, topology)."""
    params = run.data.params
    tags = run.data.tags
    rho = params.get("rho", "?")
    trial = tags.get("trial", params.get("trial", "?"))
    topology = params.get("topology", "?")
    return rho, trial, topology


def find_run_by_tags(
    experiment_name: str,
    tags_dict: Dict[str, str],
) -> Optional[mlflow.entities.Run]:
    """Search for a FINISHED MLflow run matching the provided exact tags."""
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return None

    filter_parts = [f"tags.{k} = '{v}'" for k, v in tags_dict.items()]
    filter_parts.append("attributes.status = 'FINISHED'")
    filter_str = " and ".join(filter_parts)

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=filter_str,
        max_results=1,
        output_format="list",
    )
    return runs[0] if runs else None


def find_finished_run(
    experiment_name: str,
    cfg_hash_value: str,
    kind: str | None = None,
) -> Optional[mlflow.entities.Run]:
    """
    Search for a FINISHED MLflow run matching ``cfg_hash``.
    Returns the run if found, else ``None``.
    """
    tags = {"cfg_hash": cfg_hash_value}
    if kind:
        tags["kind"] = kind
    return find_run_by_tags(experiment_name, tags)


def find_finished_identity_run(
    experiment_name: str,
    kind: str,
    identity_hash_val: str,
) -> Optional[mlflow.entities.Run]:
    """Search for FINISHED run by run kind + identity_hash tag."""
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return None
    filter_str = (
        f"tags.kind = '{kind}' and "
        f"tags.identity_hash = '{identity_hash_val}' and "
        f"attributes.status = 'FINISHED'"
    )
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=filter_str,
        max_results=1,
        output_format="list",
    )
    return runs[0] if runs else None


def find_finished_model_run(
    experiment_name: str,
    cfg: DictConfig,
    seed: int,
) -> tuple[Optional[mlflow.entities.Run], str]:
    """Find a FINISHED model run matching the given config.

    Returns (run_or_None, identity_hash_str).  Scripts should use this instead
    of computing model_identity_fields + identity_hash themselves.
    """
    fields = _model_identity_fields(cfg, seed)
    hash_val = _identity_hash("model", **fields)
    return find_finished_identity_run(experiment_name, "model", hash_val), hash_val


def load_finished_model(
    experiment_name: str,
    cfg: DictConfig,
    seed: int,
):
    """Find a FINISHED model run and load its best weights.

    Returns (model_or_None, run_id_or_None).  Drop-in replacement for the
    old cfg_hash-based get_existing_model(); uses identity hash internally.
    """
    run, _ = find_finished_model_run(experiment_name, cfg, seed)
    if run is None:
        return None, None
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/e2e_best"
    print(f"Loading model weights from {model_uri}...")
    loaded_model = mlflow.pytorch.load_model(model_uri)
    return loaded_model, run_id


# ───────────────── common tags ─────────────────


def model_tags(
    cfg: DictConfig,
    cfg_hash_value: str,
) -> Dict[str, str]:
    """Standard tag dict for a *model* training run."""
    return {
        "kind": "model",
        "schema_version": str(cfg.schema_version),
        "cfg_hash": cfg_hash_value,
        "trial": str(cfg.trial),
    }


def behaviour_tags(
    *,
    kind: str,
    behaviour: str,
    component_run_ids: list[str],
    behaviour_input_hash: str,
    component_set_hash: str,
    rho: str | None = None,
    extra: Dict[str, str] | None = None,
) -> Dict[str, str]:
    """Standard tag dict for a *behaviour* (ensemble / meta-learner) run."""
    tags = {
        "kind": kind,
        "component_set_hash": component_set_hash,
        "behaviour_input_hash": behaviour_input_hash,
        "identity_hash": behaviour_input_hash,
    }
    if rho is not None:
        tags["rho"] = rho
    if extra:
        tags.update(extra)
    return tags


def component_set_hash(run_ids: list[str]) -> str:
    return _component_set_hash(run_ids)


# ───────────────── per-step idempotency ─────────────────


def get_inference_run(
    experiment_id_or_name: str | list[str],
    trained_model_run_id: str,
    split: str = "test",
) -> pd.DataFrame:
    """
    Fetch the corresponding FINISHED inference run for a given model run and split.
    Returns a pandas DataFrame of matching runs.
    """
    if isinstance(experiment_id_or_name, str):
        exp = mlflow.get_experiment_by_name(experiment_id_or_name)
        if exp is None:
            return pd.DataFrame()
        experiment_ids = [exp.experiment_id]
    else:
        # Fallback to iterable of IDs
        experiment_ids = list(experiment_id_or_name)

    filter_str = (
        f"tags.kind = 'inference' and "
        f"tags.trained_model_run_id = '{trained_model_run_id}' and "
        f"params.split = '{split}' and "
        f"attributes.status = 'FINISHED'"
    )
    return mlflow.search_runs(
        experiment_ids=experiment_ids,
        filter_string=filter_str,
    )


def find_finished_ensemble_run(
    experiment_name: str,
    identity_hash_val: str,
    ensemble_method: str = "",
) -> Optional[mlflow.entities.Run]:
    """Check if an ensemble run already exists."""
    return find_finished_identity_run(experiment_name, "ensemble", identity_hash_val)


def find_finished_metalearner_run(
    experiment_name: str,
    identity_hash_val: str,
    meta_type: str = "",
) -> Optional[mlflow.entities.Run]:
    """Check if a metalearner run already exists."""
    return find_finished_identity_run(experiment_name, "metalearner", identity_hash_val)


def find_finished_diagnostic_run(
    experiment_name: str, identity_hash_val: str
) -> Optional[mlflow.entities.Run]:
    """Check if a diagnostic run already exists."""
    return find_finished_identity_run(experiment_name, "diagnostics", identity_hash_val)


def find_finished_diversity_run(
    experiment_name: str,
    identity_hash_val: str,
) -> Optional[mlflow.entities.Run]:
    """Check if a diversity run already exists."""
    return find_finished_identity_run(experiment_name, "diversity", identity_hash_val)


def find_finished_consistency_run(
    experiment_name: str, identity_hash_val: str
) -> Optional[mlflow.entities.Run]:
    """Check if a consistency run already exists for this hash."""
    return find_finished_identity_run(experiment_name, "consistency", identity_hash_val)


# ───────────────── category similarity profile ─────────────────


def category_similarity_profile_tags(
    parent_run_id: str,
    anchor_spec_hash: str,
    identity_hash: str,
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
        "identity_hash": identity_hash,
        "similarity_metric": similarity_metric,
        "split": split,
        "profile_hash": profile_hash,
    }
    if extra:
        tags.update(extra)
    return tags


def find_finished_similarity_profile_run(
    experiment_name: str,
    identity_hash_val: str,
) -> Optional[mlflow.entities.Run]:
    """Check if a category_similarity_profile run already exists."""
    return find_finished_identity_run(
        experiment_name, "category_similarity_profile", identity_hash_val
    )
