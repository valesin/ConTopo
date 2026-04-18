"""Example functional (no-object) service for run/artifact access.

This module shows a class-free alternative to a singleton service.
State is held at module scope, initialized once via `configure_service`.
"""

from __future__ import annotations

from threading import Lock
from typing import Optional, cast

import mlflow
from mlflow.entities import Run

_STATE_LOCK = Lock()
_STATE = {
    "configured": False,
    "tracking_uri": None,
    "experiment_name": None,
    "experiment_id": None,
}


def configure_service(
    tracking_uri: str = "sqlite:///outputs/mlflow.db",
    experiment_name: str = "contopo",
) -> None:
    """Initialize module state once.

    Calling it multiple times with the same values is harmless.
    Calling with different values raises to prevent silent reconfiguration.
    """
    with _STATE_LOCK:
        if _STATE["configured"]:
            if (
                _STATE["tracking_uri"] != tracking_uri
                or _STATE["experiment_name"] != experiment_name
            ):
                raise RuntimeError(
                    "Service already configured with different settings: "
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


def is_configured() -> bool:
    return bool(_STATE["configured"])


def get_experiment_id() -> str:
    _assert_configured()
    return str(_STATE["experiment_id"])


def find_finished_run_by_identity(kind: str, identity_hash: str) -> Optional[Run]:
    """Return first FINISHED run matching kind + identity hash."""
    _assert_configured()
    filter_str = (
        f"tags.kind = '{kind}' and "
        f"tags.identity_hash = '{identity_hash}' and "
        "attributes.status = 'FINISHED'"
    )
    runs = cast(
        list[Run],
        mlflow.search_runs(
            experiment_ids=[get_experiment_id()],
            filter_string=filter_str,
            max_results=1,
            output_format="list",
        ),
    )
    return runs[0] if runs else None


def build_run_artifact_uri(run_id: str, artifact_path: str) -> str:
    return f"runs:/{run_id}/{artifact_path}"


def _assert_configured() -> None:
    if not _STATE["configured"]:
        raise RuntimeError(
            "Functional run/artifact service is not configured. "
            "Call configure_service(...) first."
        )


# Example usage inside a stage runner:
#
# from src.repositories.functional_service_example import (
#     configure_service,
#     find_finished_run_by_identity,
#     build_run_artifact_uri,
# )
#
# configure_service(cfg.mlflow.tracking_uri, cfg.mlflow.experiment_name)
# run = find_finished_run_by_identity("inference", inf_identity_hash)
# if run is None:
#     raise RuntimeError("Missing inference run")
# artifact_uri = build_run_artifact_uri(run.info.run_id, "inference/test_tensors.npz")
