# Analysis retrieval helpers — two-layer rule
#
# This file is the analysis layer. It may import from src/ but is NEVER
# imported by pipeline scripts (01–05). Dependency direction is one-way:
#
#   analysis scripts → mlflow_helpers → src/ (pipeline infra)
#
# Functions here fall into three categories:
#   - Generic retrieval (get_runs)       — pandas/polars DataFrame of all runs of a kind
#   - Result functions  (get_X_results)  — normalised pandas DF ready for plotting
#   - Artifact functions (load_X)        — deserialise MLflow artifacts

import warnings
from pathlib import Path
from typing import Literal

import mlflow
import pandas as pd
import polars as pl

from src.repositories.functional_run_repository import (
    get_artifact_cache_dir,
    get_experiment_name,
    get_run,
    search_runs_by,
)
from src.mlflow_utils import load_mlflow_artifact


# ── Generic retrieval ─────────────────────────────────────────────────────────


def get_runs(
    kind: str,
    status: str = "FINISHED",
    output: Literal["pandas", "polars"] = "pandas",
    **fields,
) -> pd.DataFrame | pl.DataFrame:
    """Return all runs of a given kind, optionally filtered by field values.

    Fields are resolved against the telemetry schema to determine whether they
    are tags or params. Example::

        get_runs("model")
        get_runs("model", rho="0.1")
        get_runs("inference", split="test", trained_model_run_id="abc")
    """
    df = search_runs_by(kind, status=status, output="pandas", **fields)
    if df.empty:
        warnings.warn(f"No runs found for kind='{kind}' (status={status}).")
    if output == "polars":
        return pl.from_pandas(df)
    return df


# ── Inspection utilities ─────────────────────────────────────────────────────


def varying_fields(df: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of params.*/tags.* columns with more than one unique value.

    Columns: field, n_unique, values (comma-separated string), null_count.
    Sorted by n_unique descending. Use after get_runs() to see what varies across
    a result set. The returned DataFrame renders directly as a table in Marimo.
    """
    rows = []
    if isinstance(df, pl.DataFrame):
        candidate = [
            c for c in df.columns if c.startswith("params.") or c.startswith("tags.")
        ]
        for col in candidate:
            n = df[col].n_unique()
            if n <= 1:
                continue
            values = df[col].drop_nulls().unique().sort().to_list()
            rows.append(
                {
                    "field": col,
                    "n_unique": n,
                    "values": ", ".join(str(v) for v in values),
                    "null_count": df[col].null_count(),
                }
            )
    else:
        candidate = [
            c for c in df.columns if c.startswith("params.") or c.startswith("tags.")
        ]
        for col in candidate:
            n = df[col].nunique()
            if n <= 1:
                continue
            values = sorted(df[col].dropna().unique(), key=str)
            rows.append(
                {
                    "field": col,
                    "n_unique": n,
                    "values": ", ".join(str(v) for v in values),
                    "null_count": int(df[col].isna().sum()),
                }
            )
    result = pd.DataFrame(rows, columns=["field", "n_unique", "values", "null_count"])
    return result.sort_values("n_unique", ascending=False).reset_index(drop=True)


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


def get_ensemble_results(split: str = "test") -> pd.DataFrame:
    """
    Normalised ensemble results for the given split.

    Columns: run_id, rho, rho_numeric, cs_hash, vote_method, ensemble_name,
             accuracy, comp_mean_acc
    """
    runs = get_runs("ensemble", split=split)
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


def get_metalearner_results() -> pd.DataFrame:
    """
    Normalised metalearner results for all finished runs.

    Note: the inference split is not logged as a param in script 05 and
    therefore cannot be used as a filter here. All finished metalearner runs
    are returned regardless of the underlying split.

    Columns: run_id, rho, rho_numeric, cs_hash, meta_type, feature_type,
             similarity_metric, split_seed, ensemble_name, profile_mask, accuracy
    """
    runs = get_runs("metalearner")
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


def get_expected_component_hashes(groups_name: str) -> set[str]:
    """Compute component_set_hash values expected from a named groups config.

    Re-runs the same model discovery the pipeline uses (discover_ensembles_from_cfg),
    then hashes each expected combination. Robust to idempotency deduplication across
    different groups configs. Must be called after setup_environment().

    NOTE: When sample_size is set, adding new model runs to the pool is safe — old
    k-combinations remain valid subsets of the larger pool and their hashes are still
    returned. When sample_size is null (full-pool), adding new models changes the
    full-pool hash, making previously computed full-pool ensemble runs invisible.
    """
    from omegaconf import OmegaConf
    from src.ensemble.selector import discover_ensembles_from_cfg
    from src.config.hash import component_set_hash as _csh
    from src.config.notebook import compose_groups

    groups_cfg = compose_groups(groups_name)
    cfg = OmegaConf.create({"groups": OmegaConf.to_container(groups_cfg, resolve=True)})
    ensembles = discover_ensembles_from_cfg(cfg, get_experiment_name())
    return {_csh(run_ids) for run_ids in ensembles.values()}


def get_ensemble_results_for_groups(
    groups_name: str,
    split: str = "test",
) -> pd.DataFrame:
    """Normalized ensemble results for all runs generated by a named groups config.

    Uses component_set_hash matching — not groups_signature — so it is robust to
    idempotency deduplication across different groups configs.
    Columns: same as get_ensemble_results().
    """
    hashes = get_expected_component_hashes(groups_name)
    df = get_ensemble_results(split=split)
    if df.empty:
        return df
    return df[df["cs_hash"].isin(hashes)].copy()


# ── Artifact loaders ──────────────────────────────────────────────────────────


def load_inference_artifacts(
    run_id: str, split: str = "test"
) -> tuple[pd.DataFrame, dict]:
    """Load inference results and tensors for a run.

    Returns: (results_df as pandas DataFrame, tensors as dict of numpy arrays)
    """
    _cache = get_artifact_cache_dir()
    results_df = load_mlflow_artifact(
        run_id, f"inference/{split}_inference_results.parquet", cache_dir=_cache
    )
    tensors = load_mlflow_artifact(
        run_id, f"inference/{split}_tensors.npz", cache_dir=_cache
    )
    return results_df, tensors


def load_profile_artifacts(
    run_id: str,
    split: str = "test",
    similarity_metric: str = "cosine",
):
    """Load category-similarity profile tensor for a run.

    Returns: profile tensor (torch.Tensor), or None if not found.
    """
    return load_mlflow_artifact(
        run_id,
        f"profiles/{split}_{similarity_metric}_profiles.pt",
        cache_dir=get_artifact_cache_dir(),
        strict=False,
    )


def load_adapter_inputs(
    run_id: str,
    behaviour_input_hash: str | None = None,
) -> dict:
    """Load adapter inputs for a run.

    Returns: dict of numpy arrays (keys from the .npz archive).
    """
    if behaviour_input_hash is None:
        run = get_run(run_id)
        behaviour_input_hash = run.data.tags.get("behaviour_input_hash")
        if not behaviour_input_hash:
            raise ValueError(
                f"Run {run_id} is missing the 'behaviour_input_hash' tag. "
                "Please provide it manually."
            )
    return load_mlflow_artifact(
        run_id,
        f"inputs/adapter_inputs_{behaviour_input_hash}.npz",
        cache_dir=get_artifact_cache_dir(),
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
