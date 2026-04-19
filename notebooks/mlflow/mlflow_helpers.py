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

import warnings
import mlflow
import pandas as pd
import polars as pl
from pathlib import Path
from src.repositories.functional_run_repository import (
    search_runs,
)

# ── Raw list functions ────────────────────────────────────────────────────────
# Return the full search_runs result as a Polars DataFrame.
# Use for ad-hoc exploration, schema inspection, or joins across kinds.


def _search_kind(experiment: mlflow.entities.Experiment, kind: str) -> pl.DataFrame:
    df = pl.from_pandas(
        search_runs(
            f"tags.kind = '{kind}' and attributes.status = 'FINISHED'",
            output_format="pandas",
        )
    )
    if df.is_empty():
        warnings.warn(
            f"No runs found for kind='{kind}' in experiment '{experiment.name}'."
        )
    return df


def get_ensemble_list(experiment: mlflow.entities.Experiment) -> pl.DataFrame:
    return _search_kind(experiment, "ensemble")


def get_metalearner_list(experiment: mlflow.entities.Experiment) -> pl.DataFrame:
    return _search_kind(experiment, "metalearner")


def get_base_model_list(experiment: mlflow.entities.Experiment) -> pl.DataFrame:
    return _search_kind(experiment, "model")


def get_category_similarity_list(
    experiment: mlflow.entities.Experiment,
) -> pl.DataFrame:
    return _search_kind(experiment, "category_similarity_profile")


def get_inference_list(experiment: mlflow.entities.Experiment) -> pl.DataFrame:
    return _search_kind(experiment, "inference")


def get_consistency_list(experiment: mlflow.entities.Experiment) -> pl.DataFrame:
    return _search_kind(experiment, "consistency")


def get_diversity_list(experiment: mlflow.entities.Experiment) -> pl.DataFrame:
    return _search_kind(experiment, "diversity")


def get_diagnostic_list(experiment: mlflow.entities.Experiment) -> pl.DataFrame:
    return _search_kind(experiment, "diagnostics")


# ── Inspection utilities ─────────────────────────────────────────────────────


def varying_fields(df: pl.DataFrame) -> list[str]:
    """Return params.* and tags.* columns with more than one unique value and print them.

    Use after a raw list function (get_X_list) to detect runs from different
    configurations mixed in the same result set.
    """
    candidate = [
        c for c in df.columns if c.startswith("params.") or c.startswith("tags.")
    ]
    fields = [c for c in candidate if df[c].n_unique() > 1]
    if not fields:
        print("No varying params/tags — all runs share identical field values.")
        return fields
    print(f"{len(fields)} varying field(s):\n")
    for col in fields:
        values = df[col].drop_nulls().unique().sort().to_list()
        null_count = df[col].null_count()
        values_str = ", ".join(str(v) for v in values)
        null_note = f"  (+{null_count} null)" if null_count else ""
        print(f"  {col}\n    [{values_str}]{null_note}\n")
    return fields


# ── Result functions ──────────────────────────────────────────────────────────
# Return normalised pandas DataFrames with renamed columns, rho_numeric, and
# consistent sort order. Use these when aggregating or plotting results.


def _coalesce_rho(runs: pd.DataFrame) -> pd.DataFrame:
    """Ensure params.rho exists by falling back to tags.rho."""
    if "params.rho" not in runs.columns and "tags.rho" in runs.columns:
        runs = runs.copy()
        runs["params.rho"] = runs["tags.rho"]
    elif "params.rho" in runs.columns and "tags.rho" in runs.columns:
        runs = runs.copy()
        runs["params.rho"] = runs["params.rho"].fillna(runs["tags.rho"])
    return runs


def get_ensemble_results(
    experiment_name: str,
    split: str = "test",
) -> pd.DataFrame:
    """
    Normalised ensemble results for the given split.

    Columns: run_id, rho, rho_numeric, cs_hash, vote_method, ensemble_name,
             accuracy, comp_mean_acc
    """
    filter_str = (
        f"tags.kind = 'ensemble' and "
        f"params.split = '{split}' and "
        f"attributes.status = 'FINISHED'"
    )
    runs = search_runs(filter_str, output_format="pandas")
    if runs.empty:
        return pd.DataFrame()

    runs = _coalesce_rho(runs)
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
    filter_str = "tags.kind = 'metalearner' and " "attributes.status = 'FINISHED'"
    runs = search_runs(filter_str, output_format="pandas")
    if runs.empty:
        return pd.DataFrame()

    runs = _coalesce_rho(runs)
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


def get_expected_component_hashes(groups_name: str, experiment_name: str) -> set[str]:
    """Compute component_set_hash values expected from a named groups config.

    Re-runs the same model discovery the pipeline uses (discover_ensembles_from_cfg),
    then hashes each expected combination. Robust to idempotency deduplication across
    different groups configs. Must be called after setup_environment().
    """
    from omegaconf import OmegaConf
    from src.ensemble.selector import discover_ensembles_from_cfg
    from src.config.hash import component_set_hash as _csh
    from src.config.notebook import compose_groups

    groups_cfg = compose_groups(groups_name)
    cfg = OmegaConf.create({"groups": OmegaConf.to_container(groups_cfg, resolve=True)})
    ensembles = discover_ensembles_from_cfg(cfg, experiment_name)
    return {_csh(run_ids) for run_ids in ensembles.values()}


def get_ensemble_results_for_groups(
    groups_name: str,
    experiment: mlflow.entities.Experiment,
    split: str = "test",
) -> pd.DataFrame:
    """Normalized ensemble results for all runs generated by a named groups config.

    Uses component_set_hash matching — not groups_signature — so it is robust to
    idempotency deduplication across different groups configs.
    Columns: same as get_ensemble_results().
    """
    hashes = get_expected_component_hashes(groups_name, experiment.name)
    df = get_ensemble_results(experiment.name, split=split)
    if df.empty:
        return df
    return df[df["cs_hash"].isin(hashes)].copy()


def get_inference_run(
    experiment_name: str,
    trained_model_run_id: str,
    split: str = "test",
) -> pd.DataFrame:
    """Return FINISHED inference runs for a parent model run and split."""
    filter_str = (
        f"tags.kind = 'inference' and "
        f"tags.trained_model_run_id = '{trained_model_run_id}' and "
        f"params.split = '{split}' and "
        f"attributes.status = 'FINISHED'"
    )
    return search_runs(filter_str, output_format="pandas")


def get_profile_run(
    experiment_name: str,
    parent_run_id: str,
    similarity_metric: str,
    split: str = "test",
) -> pd.DataFrame:
    """Return FINISHED category_similarity_profile runs for a model/metric/split."""
    filter_str = (
        "tags.kind = 'category_similarity_profile' and "
        f"tags.parent_run_id = '{parent_run_id}' and "
        f"tags.similarity_metric = '{similarity_metric}' and "
        f"tags.split = '{split}' and "
        "attributes.status = 'FINISHED'"
    )
    return search_runs(filter_str, output_format="pandas")


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
    run_id: str,
    artifact_path: str = "inference",
    dst_path: str | None = None,
    split: str = "test",
) -> dict:
    """Download inference artifacts and return local paths dict."""
    local_dir = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path, dst_path=dst_path
    )
    p = Path(local_dir)
    result = {"local_dir": str(local_dir)}
    if (p / f"{split}_inference_results.parquet").exists():
        result["parquet"] = str(p / f"{split}_inference_results.parquet")
    if (p / f"{split}_tensors.npz").exists():
        result["npz"] = str(p / f"{split}_tensors.npz")
    return result


def load_inference_results(
    run_id: str, artifact_path: str = "inference", split: str = "test"
) -> tuple[pl.DataFrame, dict]:
    """Download and load inference artifacts. Returns (results_df, tensors_dict)."""
    import numpy as _np

    paths = download_inference_artifacts(
        run_id, artifact_path=artifact_path, split=split
    )
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
        run_id=run_id,
        split=split,
        similarity_metric=similarity_metric,
        artifact_path=artifact_path,
    )
    profile_tensor = None
    if "tensor_pt" in paths:
        profile_tensor = _torch.load(paths["tensor_pt"], map_location="cpu")
    return paths, profile_tensor


def download_adapter_inputs(
    run_id: str,
    behaviour_input_hash: str | None = None,
    artifact_path: str = "inputs",
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
    artifact_path: str = "inputs",
) -> tuple[dict, dict]:
    """Download and load adapter inputs. Returns (paths_dict, inputs_dict)."""
    import numpy as _np

    paths = download_adapter_inputs(
        run_id=run_id,
        behaviour_input_hash=behaviour_input_hash,
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
    artifact_path: str = "inference",
) -> tuple[pl.DataFrame, dict]:
    """Find the FINISHED inference run for a model and load its data."""
    filter_string = (
        f"tags.kind = 'inference' and "
        f"tags.trained_model_run_id = '{trained_model_run_id}' and "
        f"params.split = '{split}' and "
        f"attributes.status = 'FINISHED'"
    )
    runs_pd = search_runs(filter_string, output_format="pandas")
    if runs_pd.empty:
        raise ValueError(
            f"No FINISHED inference run found for model {trained_model_run_id} "
            f"on split '{split}'."
        )
    return load_inference_results(
        run_id=runs_pd.iloc[0].run_id, artifact_path=artifact_path, split=split
    )


# ── Metric history ───────────────────────────────────────────────────────────


def get_metric_history(run_id: str, metric_key: str) -> pd.DataFrame:
    """Return per-step history of *metric_key* for a single run.

    Columns: step, value, timestamp_ms
    """
    client = mlflow.tracking.MlflowClient()
    history = client.get_metric_history(run_id, metric_key)
    if not history:
        return pd.DataFrame(columns=["step", "value", "timestamp_ms"])
    return pd.DataFrame(
        [
            {"step": m.step, "value": m.value, "timestamp_ms": m.timestamp}
            for m in history
        ]
    )


# ── Plot saving ──────────────────────────────────────────────────────────────

_DEFAULT_PLOT_DIR = Path("notebooks/mlflow/saved_plots")


def save_plot(
    fig,
    name: str,
    directory: str | Path = _DEFAULT_PLOT_DIR,
    include_plotlyjs: str = "cdn",
) -> Path:
    """Save a Plotly figure as a standalone HTML file.

    Returns the path to the saved file.
    """
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    if not name.endswith(".html"):
        name = f"{name}.html"
    path = d / name
    fig.write_html(
        str(path), include_plotlyjs=include_plotlyjs, full_html=True, auto_open=False
    )
    print(f"Plot saved: {path}")
    return path
