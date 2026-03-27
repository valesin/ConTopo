# Analysis retrieval helpers — two-layer rule
#
# This file is the analysis layer. It may import from src/ but is NEVER
# imported by pipeline scripts (01–05). Dependency direction is one-way:
#
#   analysis scripts → mlflow_helpers → src/mlflow_utils (pipeline infra)
#
# Functions here fall into three categories:
#   - Raw list functions  (get_X_list)    — full Polars DF of all runs of a kind
#   - Result functions    (get_X_results) — normalised pandas DF ready for plotting
#   - Artifact functions  (load_X / download_X) — deserialise MLflow artifacts

import mlflow
import pandas as pd
import polars as pl
from pathlib import Path


# ── Raw list functions ────────────────────────────────────────────────────────
# Return the full search_runs result as a Polars DataFrame.
# Use for ad-hoc exploration, schema inspection, or joins across kinds.


def get_ensemble_list(experiment: mlflow.entities.Experiment) -> pl.DataFrame:
    return pl.from_pandas(
        mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.kind = 'ensemble'",
        )
    )


def get_metalearner_list(experiment: mlflow.entities.Experiment) -> pl.DataFrame:
    return pl.from_pandas(
        mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.kind = 'metalearner'",
        )
    )


def get_base_model_list(experiment: mlflow.entities.Experiment) -> pl.DataFrame:
    return pl.from_pandas(
        mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.kind = 'model'",
        )
    )


def get_category_similarity_list(
    experiment: mlflow.entities.Experiment,
) -> pl.DataFrame:
    return pl.from_pandas(
        mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.kind = 'category_similarity_profile'",
        )
    )


def get_inference_list(experiment: mlflow.entities.Experiment) -> pl.DataFrame:
    return pl.from_pandas(
        mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.kind = 'inference'",
        )
    )


# ── Result functions ──────────────────────────────────────────────────────────
# Return normalised pandas DataFrames with renamed columns, rho_numeric, and
# consistent sort order. Use these when aggregating or plotting results.


def get_ensemble_results(
    experiment_name: str,
    split: str = "test",
) -> pd.DataFrame:
    """
    Normalised ensemble results for the given split.

    Columns: run_id, rho, rho_numeric, cs_hash, vote_method, ensemble_name,
             accuracy, comp_mean_acc
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

    cols = {
        "run_id": "run_id",
        "params.rho": "rho",
        "tags.component_set_hash": "cs_hash",
        "params.method": "vote_method",
        "tags.ensemble_name": "ensemble_name",
        "metrics.ensemble_accuracy": "accuracy",
        "metrics.comp_mean_acc": "comp_mean_acc",
    }
    df = runs.rename(columns=cols)
    keep = [c for c in cols.values() if c in df.columns]
    result = df[keep].copy()

    if "rho" in result.columns:
        result["rho_numeric"] = pd.to_numeric(result["rho"], errors="coerce")
        result = result.sort_values(["rho_numeric", "vote_method"])

    return result


def get_metalearner_results(
    experiment_name: str,
) -> pd.DataFrame:
    """
    Normalised metalearner results for all finished runs.

    Note: the inference split is not logged as a param in script 05 and
    therefore cannot be used as a filter here. All finished metalearner runs
    are returned regardless of the underlying split.

    Columns: run_id, rho, rho_numeric, cs_hash, meta_type, feature_type,
             similarity_metric, split_seed, ensemble_name, profile_mask, accuracy
    """
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return pd.DataFrame()

    filter_str = (
        "tags.kind = 'metalearner' and "
        "attributes.status = 'FINISHED'"
    )
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=filter_str,
    )
    if runs.empty:
        return pd.DataFrame()

    cols = {
        "run_id": "run_id",
        "params.rho": "rho",
        "tags.component_set_hash": "cs_hash",
        "params.meta_type": "meta_type",
        "params.feature_type": "feature_type",
        "params.similarity_metric": "similarity_metric",
        "params.meta_split_seed": "split_seed",
        "tags.ensemble_name": "ensemble_name",
        "params.profile_mask": "profile_mask",
        "metrics.holdout_acc": "accuracy",
    }
    df = runs.rename(columns=cols)
    keep = [c for c in cols.values() if c in df.columns]
    result = df[keep].copy()

    if "rho" in result.columns:
        result["rho_numeric"] = pd.to_numeric(result["rho"], errors="coerce")
        result = result.sort_values(["rho_numeric", "meta_type", "feature_type"])

    return result


# ── Artifact functions ────────────────────────────────────────────────────────


def get_inference_artifacts(run_id: str) -> pl.DataFrame:
    """List artifacts for a run and return as a Polars DataFrame."""
    infos = mlflow.artifacts.list_artifacts(run_id=run_id)
    rows = [
        {"path": i.path, "is_dir": i.is_dir, "file_size": getattr(i, "file_size", None)}
        for i in infos
    ]
    if not rows:
        return pl.DataFrame(
            [], schema={"path": pl.Utf8, "is_dir": pl.Boolean, "file_size": pl.Int64}
        )
    return pl.DataFrame(rows)


def get_run_artifact_uri(run_id: str) -> str:
    """Return the artifact URI for a run."""
    return mlflow.get_run(run_id).info.artifact_uri


def download_inference_artifacts(
    run_id: str, artifact_path: str = "inference_data", dst_path: str | None = None
) -> dict:
    """Download inference_data artifacts and return local paths dict."""
    local_dir = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path, dst_path=dst_path
    )
    p = Path(local_dir)
    result = {"local_dir": str(local_dir)}
    if (p / "test_inference_results.parquet").exists():
        result["parquet"] = str(p / "test_inference_results.parquet")
    if (p / "test_tensors.npz").exists():
        result["npz"] = str(p / "test_tensors.npz")
    return result


def load_inference_results(
    run_id: str, artifact_path: str = "inference_data"
) -> tuple[pl.DataFrame, dict]:
    """Download and load inference artifacts. Returns (results_df, tensors_dict)."""
    import numpy as _np

    paths = download_inference_artifacts(run_id, artifact_path=artifact_path)
    results_df = pl.read_parquet(paths["parquet"]) if "parquet" in paths else None
    tensors: dict = {}
    if "npz" in paths:
        with _np.load(paths["npz"]) as d:
            tensors = {k: d[k] for k in d.files}
    return results_df, tensors


def download_profile_artifacts(
    run_id: str,
    split: str = "test",
    similarity_metric: str = "cosine",
    artifact_path: str = "profiles",
    dst_path: str | None = None,
) -> dict:
    """Download profile artifacts and return local paths dict."""
    local_dir = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path, dst_path=dst_path
    )
    p = Path(local_dir)
    result = {"local_dir": str(local_dir)}
    profile_pt = p / f"{split}_{similarity_metric}_profiles.pt"
    if profile_pt.exists():
        result["tensor_pt"] = str(profile_pt)
    return result


def load_profile_results(
    run_id: str,
    split: str = "test",
    similarity_metric: str = "cosine",
    artifact_path: str = "profiles",
) -> tuple[dict, object | None]:
    """Download and load category-similarity profile artifacts. Returns (paths_dict, tensor)."""
    import torch as _torch

    paths = download_profile_artifacts(
        run_id=run_id, split=split, similarity_metric=similarity_metric,
        artifact_path=artifact_path,
    )
    profile_tensor = None
    if "tensor_pt" in paths:
        profile_tensor = _torch.load(paths["tensor_pt"], map_location="cpu")
    return paths, profile_tensor


def download_adapter_inputs(
    run_id: str,
    behaviour_input_hash: str | None = None,
    artifact_path: str = "adapter_inputs",
    dst_path: str | None = None,
) -> dict:
    """Download adapter inputs and return local paths dict."""
    if behaviour_input_hash is None:
        run = mlflow.get_run(run_id)
        behaviour_input_hash = run.data.tags.get("behaviour_input_hash")
        if not behaviour_input_hash:
            raise ValueError(
                f"Run {run_id} is missing the 'behaviour_input_hash' tag. "
                "Please provide it manually."
            )
    local_dir = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path, dst_path=dst_path
    )
    p = Path(local_dir)
    result = {"local_dir": str(local_dir)}
    inputs_npz = p / f"adapter_inputs_{behaviour_input_hash}.npz"
    if inputs_npz.exists():
        result["npz"] = str(inputs_npz)
    return result


def load_adapter_inputs(
    run_id: str,
    behaviour_input_hash: str | None = None,
    artifact_path: str = "adapter_inputs",
) -> tuple[dict, dict]:
    """Download and load adapter inputs. Returns (paths_dict, inputs_dict)."""
    import numpy as _np

    paths = download_adapter_inputs(
        run_id=run_id, behaviour_input_hash=behaviour_input_hash,
        artifact_path=artifact_path,
    )
    inputs_dict: dict = {}
    if "npz" in paths:
        with _np.load(paths["npz"]) as d:
            inputs_dict = {k: d[k] for k in d.files}
    return paths, inputs_dict


def load_inference_results_from_model_run_id(
    experiment: mlflow.entities.Experiment,
    trained_model_run_id: str,
    split: str = "test",
    artifact_path: str = "inference_data",
) -> tuple[pl.DataFrame, dict]:
    """Find the FINISHED inference run for a model and load its data."""
    filter_string = (
        f"tags.kind = 'inference' and "
        f"tags.trained_model_run_id = '{trained_model_run_id}' and "
        f"tags.split = '{split}' and "
        f"attributes.status = 'FINISHED'"
    )
    runs_pd = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
    )
    if runs_pd.empty:
        raise ValueError(
            f"No FINISHED inference run found for model {trained_model_run_id} "
            f"on split '{split}'."
        )
    return load_inference_results(
        run_id=runs_pd.iloc[0].run_id, artifact_path=artifact_path
    )
