from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import tempfile
from typing import Any, Mapping

import mlflow

# ───────────────── timed upload helpers ─────────────────


@contextmanager
def _timed_log(description: str):
    """Context manager that prints wall-clock start/end timestamps around an MLflow upload."""
    start = datetime.now()
    print(f"[UPLOAD START] {start.strftime('%H:%M:%S')} — {description}")
    try:
        yield
    finally:
        end = datetime.now()
        elapsed = (end - start).total_seconds()
        print(
            f"[UPLOAD  END ] {end.strftime('%H:%M:%S')} — {description} "
            f"completed in {elapsed:.1f}s"
        )


def timed_log_metrics(metrics: dict, **kwargs) -> None:
    """Wrapper around ``mlflow.log_metrics`` with timing output."""
    step = kwargs.get("step", "")
    desc = f"Logging metrics (step={step})" if step != "" else "Logging metrics"
    with _timed_log(desc):
        mlflow.log_metrics(metrics, **kwargs)


def timed_log_metric(key: str, value: float, **kwargs) -> None:
    """Wrapper around ``mlflow.log_metric`` with timing output."""
    with _timed_log(f"Logging metric: {key}={value}"):
        mlflow.log_metric(key, value, **kwargs)


def timed_log_artifact(local_path: str, **kwargs) -> None:
    """Wrapper around ``mlflow.log_artifact`` with timing output."""
    import os

    artifact_path = kwargs.get("artifact_path", "")
    fname = os.path.basename(local_path)
    desc = (
        f"Logging artifact: {artifact_path}/{fname}"
        if artifact_path
        else f"Logging artifact: {fname}"
    )
    with _timed_log(desc):
        mlflow.log_artifact(local_path, **kwargs)


def timed_log_model(model, **kwargs) -> None:
    """Wrapper around ``mlflow.pytorch.log_model`` with timing output."""
    # MLflow 3.x routes ALL log_model calls (both `name` and `artifact_path`) through
    # a global models namespace (mlruns/models/m-{uuid}/), breaking the
    # runs:/{run_id}/{name} URI used by all downstream load calls.
    # Workaround: save the model locally with save_model, then upload the directory
    # as a plain artifact so it lands at runs:/{run_id}/artifacts/{name}/ as expected.
    name = kwargs.pop("name", "model")
    signature = kwargs.pop("signature", None)
    with _timed_log(f"Logging PyTorch model: {name}"):
        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow.pytorch.save_model(
                model, tmpdir, pip_requirements=[], signature=signature
            )
            mlflow.log_artifacts(tmpdir, artifact_path=name)
    # Flush async artifact uploads so the model is fully on S3 before the run
    # is marked FINISHED — a failure here propagates and marks the run FAILED.
    mlflow.flush_artifact_async_logging()


ALLOWED_PARAMS: dict[str, set[str]] = {
    "model": {
        "rho",
        "seed",
        "epochs",
        "batch_size",
        "learning_rate",
        "optimiser",
        "weight_decay",
        "momentum",
        "scheduler",
        "amp",
        "save_freq_epochs",
        "early_stopping_patience",
        "early_stopping_method",
        "beta",
        "eps",
        "lambda_max",
        "topography_type",
        "topology",
        "neighbourhood_type",
        "neighbourhood_radius",
        "embedding_dim",
        "p_dropout",
        "head_bias",
        "model_arch",
        "dataset",
        "transforms_preset",
        "split_strategy",
        "val_per_class",
    },
    "inference": {
        "dataset",
        "split",
        "transforms_preset",
        "rho",
    },
    "category_similarity_profile": {
        "similarity_metric",
        "split",
        "profile_hash",
        "num_anchors",
        "num_samples",
        "rho",
        "trial",
        "topology",
    },
    "diagnostics": {
        "diagnostic_metric",
        "split",
    },
    "ensemble": {
        "method",
        "method_type",
        "num_components",
        "split",
        "rho",
    },
    "diversity": {
        "num_components",
        "split",
        "diversity_metric",
    },
    "consistency": {
        "num_components",
        "split",
        "anchors_per_class",
    },
    "metalearner": {
        "meta_type",
        "feature_type",
        "similarity_metric",
        "adapter_epochs",
        "adapter_lr",
        "adapter_batch_size",
        "meta_split_seed",
        "meta_split_train",
        "meta_split_val",
        "adapter_architecture",
        "standardization_applied",
        "num_components",
        "profile_mask",
    },
}

ALLOWED_TAGS: dict[str, set[str]] = {
    "model": {
        "kind",
        "identity_hash",
        "schema_version",
        "cfg_hash",
        "trial",
        "run_name",
    },
    "inference": {
        "kind",
        "identity_hash",
        "trained_model_run_id",
        "parent_run_name",
        "cfg_hash",
    },
    "category_similarity_profile": {
        "kind",
        "identity_hash",
        "parent_run_id",
        "inference_run_id",
        "anchor_spec_hash",
        "profile_dim",
        "run_name",
        "profile_hash",
        "similarity_metric",
        "split",
    },
    "diagnostics": {
        "kind",
        "identity_hash",
        "parent_run_id",
        "inference_run_id",
        "run_name",
    },
    "ensemble": {
        "kind",
        "identity_hash",
        "ensemble_name",
        "component_set_hash",
        "behaviour_input_hash",
        "component_run_ids_csv",
        "feature_type",
        "rho",
    },
    "diversity": {
        "kind",
        "identity_hash",
        "ensemble_name",
        "component_set_hash",
        "run_name",
    },
    "consistency": {
        "kind",
        "identity_hash",
        "ensemble_name",
        "component_set_hash",
        "consistency_hash",
        "anchor_spec_hash",
        "run_name",
    },
    "metalearner": {
        "kind",
        "identity_hash",
        "ensemble_name",
        "component_set_hash",
        "behaviour_input_hash",
        "component_run_ids_csv",
        "run_name",
        "rho",
    },
}


def _ensure_known_kind(kind: str) -> None:
    if kind not in ALLOWED_PARAMS or kind not in ALLOWED_TAGS:
        raise ValueError(f"Unknown MLflow run kind: {kind}")


def _check_keys(
    kind: str, payload: Mapping[str, Any], allowed: set[str], slot: str
) -> None:
    unknown = sorted(set(payload.keys()) - allowed)
    if unknown:
        raise ValueError(
            f"Unsupported {slot} key(s) for kind='{kind}': {unknown}. "
            f"Allowed: {sorted(allowed)}"
        )


def _clean_params(params: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        out[key] = value
    return out


def _clean_tags(tags: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in tags.items():
        if value is None:
            continue
        out[key] = str(value)
    return out


def start_run(kind: str, run_name: str, tags: Mapping[str, Any] | None = None):
    _ensure_known_kind(kind)
    merged_tags = {"kind": kind, **(tags or {})}
    _check_keys(kind, merged_tags, ALLOWED_TAGS[kind], "tag")
    start = datetime.now()
    print(
        f"[UPLOAD START] {start.strftime('%H:%M:%S')} — Starting MLflow run: {run_name} (kind={kind})"
    )
    result = mlflow.start_run(run_name=run_name, tags=_clean_tags(merged_tags))
    end = datetime.now()
    elapsed = (end - start).total_seconds()
    print(
        f"[UPLOAD  END ] {end.strftime('%H:%M:%S')} — Starting MLflow run: {run_name} (kind={kind}) completed in {elapsed:.1f}s"
    )
    return result


def log_params(kind: str, params: Mapping[str, Any]) -> None:
    _ensure_known_kind(kind)
    cleaned = _clean_params(params)
    _check_keys(kind, cleaned, ALLOWED_PARAMS[kind], "param")
    if cleaned:
        with _timed_log(f"Logging params (kind={kind}, n={len(cleaned)})"):
            mlflow.log_params(cleaned)


def log_tags(kind: str, tags: Mapping[str, Any]) -> None:
    _ensure_known_kind(kind)
    cleaned = _clean_tags(tags)
    _check_keys(kind, cleaned, ALLOWED_TAGS[kind], "tag")
    if cleaned:
        with _timed_log(f"Logging tags (kind={kind}, n={len(cleaned)})"):
            mlflow.set_tags(cleaned)
