"""Functional run repository (no classes).

Provides explicit, module-level functions for MLflow run lookup using a
single configured experiment context.
"""

from __future__ import annotations

from threading import Lock
from typing import Any, Literal, Optional, TypedDict

import mlflow
from mlflow.entities import Run
from src.config.hash import (
    identity_hash as _identity_hash,
    model_identity_fields as _model_identity_fields,
)
from src.mlflow_schema_logger import field_mlflow_prefix as _field_mlflow_prefix

_STATE_LOCK = Lock()


class _RepositoryState(TypedDict):
    configured: bool
    tracking_uri: Optional[str]
    experiment_name: Optional[str]
    experiment_id: Optional[str]
    artifact_cache_dir: Optional[str]


_STATE: _RepositoryState = {
    "configured": False,
    "tracking_uri": None,
    "experiment_name": None,
    "experiment_id": None,
    "artifact_cache_dir": None,
}


def configure_run_repository(
    tracking_uri: str,
    experiment_name: str,
) -> None:
    """Configure repository context once.

    Repeated calls with identical values are no-ops.
    Reconfiguration with different values in the same process is rejected.
    """
    with _STATE_LOCK:
        if _STATE["configured"]:
            if (
                _STATE["tracking_uri"] != tracking_uri
                or _STATE["experiment_name"] != experiment_name
            ):
                raise RuntimeError(
                    "Run repository already configured with different values: "
                    f"tracking_uri={_STATE['tracking_uri']}, "
                    f"experiment_name={_STATE['experiment_name']}"
                )
            return

        mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        _STATE["tracking_uri"] = tracking_uri
        _STATE["experiment_name"] = experiment_name
        _STATE["experiment_id"] = experiment.experiment_id
        _STATE["configured"] = True


def ensure_run_repository(
    experiment_name: str,
    tracking_uri: str | None = None,
) -> None:
    """Ensure repository is configured for the target experiment.

    If tracking_uri is omitted, uses current MLflow tracking URI.
    """
    if tracking_uri is None:
        tracking_uri = mlflow.get_tracking_uri()
    configure_run_repository(tracking_uri=tracking_uri, experiment_name=experiment_name)


def get_experiment_id() -> str:
    _assert_configured()
    return str(_STATE["experiment_id"])


def get_experiment_name() -> str:
    _assert_configured()
    return str(_STATE["experiment_name"])


def configure_artifact_cache_dir(cache_dir: str) -> None:
    with _STATE_LOCK:
        _STATE["artifact_cache_dir"] = cache_dir


def get_artifact_cache_dir() -> Optional[str]:
    return _STATE["artifact_cache_dir"]


def search_runs(
    filter_string: str,
    *,
    max_results: int | None = None,
    output_format: Literal["list", "pandas"] = "list",
    order_by: list[str] | None = None,
) -> Any:
    """Search runs inside configured experiment."""
    _assert_configured()
    kwargs: dict[str, Any] = {
        "experiment_ids": [get_experiment_id()],
        "filter_string": filter_string,
        "output_format": output_format,
    }
    if max_results is not None:
        kwargs["max_results"] = max_results
    if order_by:
        kwargs["order_by"] = order_by
    return mlflow.search_runs(**kwargs)


def find_finished_identity_run(kind: str, identity_hash: str) -> Optional[Run]:
    """Find first FINISHED run by kind + identity hash in configured experiment."""
    _assert_configured()
    filter_str = (
        f"tags.kind = '{kind}' and "
        f"tags.identity_hash = '{identity_hash}' and "
        "attributes.status = 'FINISHED'"
    )
    runs = search_runs(filter_str, max_results=1, output_format="list")
    return runs[0] if runs else None


def find_finished_model_run(cfg: Any, seed: int) -> tuple[Optional[Run], str]:
    """Find FINISHED model run for a config/seed pair.

    Returns ``(run_or_none, model_identity_hash)``.
    """
    fields = _model_identity_fields(cfg, seed)
    model_identity_hash = _identity_hash("model", **fields)
    run = find_finished_identity_run("model", model_identity_hash)
    return run, model_identity_hash


def find_first_finished_run(filter_clause: str) -> Optional[Run]:
    """Find first FINISHED run matching a custom additional filter clause."""
    filter_str = f"{filter_clause} and attributes.status = 'FINISHED'"
    runs = search_runs(filter_str, max_results=1, output_format="list")
    return runs[0] if runs else None


def get_run(run_id: str) -> Run:
    """Thin wrapper around mlflow.get_run for symmetry in functional API."""
    _assert_configured()
    return mlflow.get_run(run_id)


def search_runs_by(
    kind: str,
    status: str = "FINISHED",
    output: Literal["pandas", "list"] = "pandas",
    **fields: Any,
) -> Any:
    """Search runs of a given kind with optional field filters.

    Field names are resolved against TELEMETRY_SCHEMA to determine whether
    they are params or tags. When a field exists in both, params takes
    precedence. Unknown fields raise ValueError.

    Example::

        search_runs_by("inference", split="test", trained_model_run_id="abc")
        search_runs_by("ensemble", rho="0.1", output="list")
    """
    clauses = [
        f"tags.kind = '{kind}'",
        f"attributes.status = '{status}'",
    ]
    for field, value in fields.items():
        prefix = _field_mlflow_prefix(kind, field)
        clauses.append(f"{prefix}.{field} = '{value}'")
    filter_string = " and ".join(clauses)
    return search_runs(filter_string, output_format=output)


def _assert_configured() -> None:
    if not _STATE["configured"]:
        raise RuntimeError(
            "Run repository is not configured. "
            "Call configure_run_repository(...) first."
        )
