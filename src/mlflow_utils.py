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
)
from src.config.paths import ensure_output_dirs

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
        mlflow.log_artifact(f.name, artifact_path="config")
        os.unlink(f.name)


def load_mlflow_artifact(
    run_id: str, artifact_path: str, file_type: str = "auto", strict: bool = False
) -> Any:
    """
    Download and load a specific artifact from an MLflow run.
    Supported types: 'auto', 'numpy', 'torch', 'parquet'.
    If strict=True, raises RuntimeError natively on failure to simplify caller scripts.
    """
    artifact_uri = f"runs:/{run_id}/{artifact_path}"
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)

    if file_type == "auto":
        if artifact_path.endswith(".npz") or artifact_path.endswith(".npy"):
            file_type = "numpy"
        elif artifact_path.endswith(".pt") or artifact_path.endswith(".pth"):
            file_type = "torch"
        elif artifact_path.endswith(".parquet"):
            file_type = "parquet"
        else:
            return local_path  # Return the downloaded path if type is unknown

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
    elif file_type == "parquet":
        try:
            return pd.read_parquet(local_path)
        except Exception as e:
            if strict:
                raise RuntimeError(
                    f"Failed to load parquet artifact {artifact_path} for run {run_id}. {e}"
                )
            raise
    else:
        raise ValueError(f"Unsupported file_type: {file_type}")


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
    mlflow.log_input(eval_dataset, context=context)


# ───────────────── idempotency ─────────────────


def resolve_seed(cfg: DictConfig) -> int:
    if cfg.seed is not None:
        return int(cfg.seed)
    return 100 + int(cfg.trial)


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


# This is a compatibility function to bridge older cfg_hash-based runs
# with the new identity_hash system, ensuring smooth transition
# without losing access to previously logged models.
# After running, all runs that had the old cfg_hash but are accessed via this function
# will be backfilled with the new identity_hash tag for future direct access.
# Delete eventually once all legacy runs have been accessed at least once and backfilled.
def find_finished_model_run_compat(
    experiment_name: str,
    identity_hash_val: str,
    cfg_hash_value: str,
) -> Optional[mlflow.entities.Run]:
    """Find a FINISHED model run with identity-hash first, cfg-hash fallback.

    If a legacy cfg-hash run is found and missing ``identity_hash``, this function
    backfills the tag once so future lookups resolve via identity hash directly.
    """
    run = find_finished_identity_run(
        experiment_name=experiment_name,
        kind="model",
        identity_hash_val=identity_hash_val,
    )
    if run is not None:
        return run

    legacy_run = find_finished_run(
        experiment_name=experiment_name,
        cfg_hash_value=cfg_hash_value,
        kind="model",
    )
    if legacy_run is None:
        return None

    existing_identity = legacy_run.data.tags.get("identity_hash")
    if existing_identity != identity_hash_val:
        client = MlflowClient()
        client.set_tag(legacy_run.info.run_id, "identity_hash", identity_hash_val)

    return legacy_run


def check_existing_model(
    experiment_name: str, cfg_hash_value: str, kind: str | None = "model"
) -> bool:
    """
    Checks if a model with this exact configuration was already trained.
    Returns True if such a model exists, False otherwise.
    """
    existing_run = find_finished_run(
        experiment_name=experiment_name, cfg_hash_value=cfg_hash_value, kind=kind
    )
    return existing_run is not None


def get_existing_model(
    experiment_name: str, cfg_hash_value: str, kind: str | None = "model"
):
    """
    Checks if a model with this exact configuration was already trained.
    If yes, downloads and returns the PyTorch model. If no, returns None.
    """
    # 1. Search for the run using your existing function
    existing_run = find_finished_run(
        experiment_name=experiment_name, cfg_hash_value=cfg_hash_value, kind=kind
    )

    # 2. If no run matches your hash, return None so your script knows to train
    if existing_run is None:
        return None

    # 3. If found, grab the Run ID
    run_id = existing_run.info.run_id
    # print(f"Found existing FINISHED run! (Run ID: {run_id})")

    # 4. Construct the MLflow Model URI pointing to the "e2e_best" artifact
    model_uri = f"runs:/{run_id}/e2e_best"

    # 5. Load the PyTorch model directly into memory
    print(f"Loading model weights from {model_uri}...")
    loaded_model = mlflow.pytorch.load_model(model_uri)

    return loaded_model, run_id


# ───────────────── common tags ─────────────────


def _format_rho(rho) -> str:
    """Consistent string representation of rho for MLflow tags."""
    return str(float(rho))


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


def get_profile_run(
    experiment_id_or_name: str | list[str],
    trained_model_run_id: str,
    similarity_metric: str,
    split: str = "test",
) -> pd.DataFrame:
    """
    Fetch the corresponding FINISHED profile run for a given model run and split.
    Returns a pandas DataFrame of matching runs.
    """
    if isinstance(experiment_id_or_name, str):
        exp = mlflow.get_experiment_by_name(experiment_id_or_name)
        if exp is None:
            return pd.DataFrame()
        experiment_ids = [exp.experiment_id]
    else:
        # Assume it's an iterable of experiment IDs
        experiment_ids = list(experiment_id_or_name)

    filter_str = (
        f"tags.kind = 'category_similarity_profile' and "
        f"tags.parent_run_id = '{trained_model_run_id}' and "
        f"params.similarity_metric = '{similarity_metric}' and "
        f"params.split = '{split}' and "
        f"attributes.status = 'FINISHED'"
    )
    return mlflow.search_runs(
        experiment_ids=experiment_ids,
        filter_string=filter_str,
    )


def get_ensemble_results(
    experiment_name: str,
    split: str = "test",
) -> pd.DataFrame:
    """
    Retrieves all ensemble evaluation runs for the given split.

    Returns a DataFrame with columns:
    [rho, cs_hash, behaviour, ensemble_name, accuracy, rho_numeric]
    """
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return pd.DataFrame()

    filter_str = (
        f"tags.kind = 'ensemble' and "
        f"params.split = '{split}' and "
        f"attributes.status = 'FINISHED'"
    )
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=filter_str,
    )

    if runs.empty:
        return pd.DataFrame()

    # Define columns to extract
    cols = {
        "run_id": "run_id",
        "params.rho": "rho",
        "tags.component_set_hash": "cs_hash",
        "params.method": "behaviour",  # vote method
        "tags.ensemble_name": "ensemble_name",
        "metrics.ensemble_accuracy": "accuracy",
    }

    df = runs.rename(columns=cols)
    keep = [c for c in cols.values() if c in df.columns]
    result = df[keep].copy()

    if "rho" in result.columns:
        result["rho_numeric"] = pd.to_numeric(result["rho"], errors="coerce")
        result = result.sort_values(["rho_numeric", "behaviour"])

    return result


def get_metalearner_results(
    experiment_name: str,
    split: str = "test",
) -> pd.DataFrame:
    """
    Retrieves all metalearner (adapter) training runs for the given split.

    Returns a DataFrame with columns:
    [rho, cs_hash, behaviour, feature_type, similarity_metric, split_seed, accuracy, rho_numeric]
    """
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return pd.DataFrame()

    filter_str = (
        f"tags.kind = 'metalearner' and "
        f"params.split = '{split}' and "
        f"attributes.status = 'FINISHED'"
    )
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=filter_str,
    )

    if runs.empty:
        return pd.DataFrame()

    # Define columns to extract
    cols = {
        "run_id": "run_id",
        "params.rho": "rho",
        "tags.component_set_hash": "cs_hash",
        "params.meta_type": "behaviour",  # meta_type
        "params.feature_type": "feature_type",
        "params.similarity_metric": "similarity_metric",
        "params.meta_split_seed": "split_seed",
        "metrics.holdout_acc": "accuracy",
    }

    df = runs.rename(columns=cols)
    keep = [c for c in cols.values() if c in df.columns]
    result = df[keep].copy()

    if "rho" in result.columns:
        result["rho_numeric"] = pd.to_numeric(result["rho"], errors="coerce")
        result = result.sort_values(["rho_numeric", "behaviour", "feature_type"])

    return result
