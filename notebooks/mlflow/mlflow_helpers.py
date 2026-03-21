import mlflow
import polars as pl
import os
from pathlib import Path


def get_ensemble_list(experiment: mlflow.entities.Experiment) -> pl.DataFrame:
    models_pd = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.kind = 'ensemble'",
    )
    return pl.from_pandas(models_pd)


def get_metalearner_list(experiment: mlflow.entities.Experiment) -> pl.DataFrame:
    models_pd = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.kind = 'metalearner'",
    )
    return pl.from_pandas(models_pd)


def get_base_model_list(experiment: mlflow.entities.Experiment) -> pl.DataFrame:
    models_pd = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.kind = 'model'",
    )
    return pl.from_pandas(models_pd)


def get_category_similarity_list(
    experiment: mlflow.entities.Experiment,
) -> pl.DataFrame:
    models_pd = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.kind = 'category_similarity_profile'",
    )
    return pl.from_pandas(models_pd)


def get_inference_list(experiment: mlflow.entities.Experiment) -> pl.DataFrame:
    models_pd = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.kind = 'inference'",
    )
    return pl.from_pandas(models_pd)


def get_inference_artifacts(run_id: str) -> pl.DataFrame:
    """List artifacts for a given run_id and return as a Polars DataFrame.

    Args:
        run_id: MLflow run ID string.

    Returns:
        pl.DataFrame with columns `path`, `is_dir`, and `file_size` (when available).
    """
    infos = mlflow.artifacts.list_artifacts(run_id=run_id)
    rows = []
    for info in infos:
        rows.append(
            {
                "path": info.path,
                "is_dir": info.is_dir,
                "file_size": getattr(info, "file_size", None),
            }
        )
    # Return an explicit empty DataFrame with the expected columns if no artifacts
    if not rows:
        return pl.DataFrame(
            [], schema={"path": pl.Utf8, "is_dir": pl.Boolean, "file_size": pl.Int64}
        )
    return pl.DataFrame(rows)


def get_run_artifact_uri(run_id: str) -> str:
    """Return the artifact URI for a run (e.g. file:///.../mlruns/0/<run_id>/artifacts)."""
    run = mlflow.get_run(run_id)
    return run.info.artifact_uri


def download_inference_artifacts(
    run_id: str, artifact_path: str = "inference_data", dst_path: str | None = None
) -> dict:
    """Download the `inference_data` artifact directory for a run and return local paths.

    Returns a dict with `local_dir` and optional `parquet`/`npz` keys when found.
    """
    local_dir = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path, dst_path=dst_path
    )
    p = Path(local_dir)
    result = {"local_dir": str(local_dir)}
    parquet_p = p / "test_inference_results.parquet"
    npz_p = p / "test_tensors.npz"
    if parquet_p.exists():
        result["parquet"] = str(parquet_p)
    if npz_p.exists():
        result["npz"] = str(npz_p)
    return result


def load_inference_results(
    run_id: str, artifact_path: str = "inference_data"
) -> tuple[pl.DataFrame, dict]:
    """Download and load expected inference artifacts.

    Returns (results_df, tensors_dict) where `results_df` is a Polars DataFrame
    loaded from `test_inference_results.parquet` (or None if missing), and
    `tensors_dict` is the loaded numpy `.npz` contents (or empty dict if missing).
    """
    import numpy as _np

    paths = download_inference_artifacts(run_id, artifact_path=artifact_path)
    results_df = None
    tensors = {}
    if "parquet" in paths:
        results_df = pl.read_parquet(paths["parquet"])
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
    """Download profile artifacts for a run and return local paths.

    Returns a dict with `local_dir` and optional `tensor_pt` key for
    `<split>_<similarity_metric>_profiles.pt` when present.
    """
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
    """Download and load category-similarity profile artifacts.

    Returns `(paths_dict, profile_tensor)` where:
      - `paths_dict` is returned by `download_profile_artifacts`
      - `profile_tensor` is loaded from `<split>_<similarity_metric>_profiles.pt`
        or `None` if missing.
    """
    import torch as _torch

    paths = download_profile_artifacts(
        run_id=run_id,
        split=split,
        similarity_metric=similarity_metric,
        artifact_path=artifact_path,
    )
    profile_tensor = None
    if "tensor_pt" in paths:
        profile_tensor = _torch.load(paths["tensor_pt"], map_location="cpu")
    return paths, profile_tensor


def setup_connection(
    root_path: str = None, port: int = None, experiment_name: str = "contopo"
) -> mlflow.entities.Experiment:
    # Prefer absolute Path objects and avoid changing the global cwd in libraries
    if root_path is not None:
        # Build absolute path to the sqlite DB (safer than relying on CWD)
        db_path = os.path.abspath(os.path.join(root_path, "outputs", "mlflow.db"))
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(
                f"Experiment '{experiment_name}' not found in MLflow database at {db_path}"
            )
        return experiment

    if port is not None:
        mlflow.set_tracking_uri(f"http://localhost:{port}")
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(
                f"Experiment '{experiment_name}' not found on MLflow server at http://localhost:{port}"
            )
        return experiment

    # At least one of root_path or port must be provided
    raise ValueError(
        "Either root_path or port must be provided to connect to MLflow database"
    )
