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


def download_adapter_inputs(
    run_id: str,
    behaviour_input_hash: str | None = None,
    artifact_path: str = "adapter_inputs",
    dst_path: str | None = None,
) -> dict:
    """Download adapter inputs for a run and return local paths.

    If `behaviour_input_hash` is not provided, it will be retrieved from
    the run's tags `tags.behaviour_input_hash`.

    Returns a dict with `local_dir` and optional `npz` key for
    `adapter_inputs_<behaviour_input_hash>.npz` when present.
    """
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
    """Download and load adapter inputs artifacts.

    If `behaviour_input_hash` is not provided, it will be automatically
    inferred from the run's tags.

    Returns `(paths_dict, inputs_dict)` where:
      - `paths_dict` is returned by `download_adapter_inputs`
      - `inputs_dict` is loaded from `adapter_inputs_<behaviour_input_hash>.npz`
        or empty dict if missing.
    """
    import numpy as _np

    paths = download_adapter_inputs(
        run_id=run_id,
        behaviour_input_hash=behaviour_input_hash,
        artifact_path=artifact_path,
    )
    inputs_dict = {}
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
    """Finds the FINISHED inference run for a given model run and loads its data.

    Returns `(results_df, tensors_dict)` exactly as `load_inference_results`.
    """
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

    inference_run_id = runs_pd.iloc[0].run_id
    return load_inference_results(run_id=inference_run_id, artifact_path=artifact_path)
