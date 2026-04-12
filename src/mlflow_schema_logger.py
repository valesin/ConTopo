from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
import tempfile
from typing import Any, Mapping

import mlflow


class TelemetryContractError(Exception):
    """Raised when an MLflow run fails to meet its telemetry schema constraints."""
    pass


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
    name = kwargs.pop("name", "model")
    signature = kwargs.pop("signature", None)
    with _timed_log(f"Logging PyTorch model: {name}"):
        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow.pytorch.save_model(
                model, tmpdir, pip_requirements=[], signature=signature
            )
            mlflow.log_artifacts(tmpdir, artifact_path=name)
    mlflow.flush_artifact_async_logging()


# ───────────────── Schema Definitions ─────────────────

TELEMETRY_SCHEMA = {
    "model": {
        "params": {
            "required": [
                "rho", "seed", "epochs", "batch_size", "learning_rate", "optimiser",
                "weight_decay", "momentum", "scheduler", "amp", "save_freq_epochs",
                "early_stopping_patience", "early_stopping_method", "beta", "eps",
                "lambda_max", "topography_type", "topology", "neighbourhood_type",
                "neighbourhood_radius", "embedding_dim", "p_dropout", "head_bias",
                "model_arch", "dataset", "transforms_preset", "split_strategy", "val_per_class"
            ],
            "optional": []
        },
        "tags": {
            "required": ["kind", "identity_hash", "schema_version", "cfg_hash", "trial"],
            "optional": ["run_name"]
        },
        "metrics": {
            "required": ["test_accuracy", "test_loss", "best_val_acc", "best_val_loss"],
            "optional": []
        },
        "artifacts": {
            "required": ["e2e_best"],
            "optional": []
        }
    },
    "inference": {
        "params": {
            "required": ["dataset", "split", "transforms_preset", "rho"],
            "optional": []
        },
        "tags": {
            "required": ["kind", "identity_hash", "trained_model_run_id", "parent_run_name", "cfg_hash"],
            "optional": ["run_name"]
        },
        "metrics": {
            "required": ["accuracy"],
            "optional": []
        },
        "artifacts": {
            "required": ["inference/{split}_inference_results.parquet", "inference/{split}_tensors.npz"],
            "optional": []
        }
    },
    "category_similarity_profile": {
        "params": {
            "required": ["similarity_metric", "split", "profile_hash", "num_anchors", "num_samples", "rho", "trial", "topology"],
            "optional": []
        },
        "tags": {
            "required": ["kind", "identity_hash", "parent_run_id", "inference_run_id", "anchor_spec_hash", "profile_dim", "profile_hash", "similarity_metric", "split"],
            "optional": ["run_name"]
        },
        "metrics": {
            "required": [],
            "optional": []
        },
        "artifacts": {
            "required": ["profiles/{split}_{similarity_metric}_profiles.pt"],
            "optional": []
        }
    },
    "diagnostics": {
        "params": {
            "required": ["diagnostic_metric", "split"],
            "optional": []
        },
        "tags": {
            "required": ["kind", "identity_hash", "parent_run_id"],
            "optional": ["run_name", "inference_run_id"]
        },
        "metrics": {
            "required": [],
            "optional": ["morans_i", "weight_norms_mean", "weight_norms_std", "unit_dist_cos_correlation"]
        },
        "artifacts": {
            "required": [],
            "optional": ["diagnostics/weight_norms.pt", "diagnostics/unit_distance_correlation.pt"]
        }
    },
    "ensemble": {
        "params": {
            "required": ["method", "method_type", "num_components", "split", "rho"],
            "optional": []
        },
        "tags": {
            "required": ["kind", "identity_hash", "ensemble_name", "component_set_hash", "behaviour_input_hash", "component_run_ids_csv", "feature_type", "rho", "groups_signature"],
            "optional": ["run_name"]
        },
        "metrics": {
            "required": ["ensemble_accuracy", "comp_mean_acc", "comp_max_acc"],
            "optional": []
        },
        "artifacts": {
            "required": ["ensemble/composition_map.json", "ensemble/{split}_{ensemble_name}_{method}_inference.parquet"],
            "optional": ["ensemble/{split}_{ensemble_name}_{method}_tensors.npz"]
        }
    },
    "diversity": {
        "params": {
            "required": ["num_components", "split", "diversity_metric"],
            "optional": []
        },
        "tags": {
            "required": ["kind", "identity_hash", "ensemble_name", "component_set_hash", "groups_signature"],
            "optional": ["run_name", "component_run_ids_csv"]
        },
        "metrics": {
            "required": [],
            "optional": ["q_statistic", "disagreement", "double_fault", "correlation", "interrater_agreement", "iou_top_n"]
        },
        "artifacts": {
            "required": [],
            "optional": []
        }
    },
    "consistency": {
        "params": {
            "required": ["num_components", "split", "anchors_per_class"],
            "optional": []
        },
        "tags": {
            "required": ["kind", "identity_hash", "ensemble_name", "component_set_hash", "consistency_hash", "anchor_spec_hash", "groups_signature"],
            "optional": ["run_name"]
        },
        "metrics": {
            "required": ["mean_rsa_correlation"],
            "optional": []
        },
        "artifacts": {
            "required": ["consistency/rsa_matrix.pt", "consistency/run_id_ordering.json"],
            "optional": []
        }
    },
    "metalearner": {
        "params": {
            "required": [
                "meta_type", "feature_type", "similarity_metric", "adapter_epochs",
                "adapter_lr", "adapter_batch_size", "meta_split_seed", "meta_split_train",
                "meta_split_val", "adapter_architecture", "standardization_applied",
                "num_components", "profile_mask"
            ],
            "optional": []
        },
        "tags": {
            "required": ["kind", "identity_hash", "ensemble_name", "component_set_hash", "behaviour_input_hash", "component_run_ids_csv", "rho", "groups_signature"],
            "optional": ["run_name"]
        },
        "metrics": {
            "required": ["val_acc", "val_loss", "holdout_acc", "holdout_loss"],
            "optional": []
        },
        "artifacts": {
            "required": [
                "inputs/adapter_inputs_{behaviour_input_hash}.npz",
                "inputs/adapter_split_trace_{behaviour_input_hash}.parquet",
                "data/adapter_holdout_{behaviour_input_hash}.parquet",
                "data/adapter_holdout_{behaviour_input_hash}.npz",
                "model"
            ],
            "optional": []
        }
    },
}


def _ensure_known_kind(kind: str) -> None:
    if kind not in TELEMETRY_SCHEMA:
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


def _check_artifact_exists(client: mlflow.tracking.MlflowClient, run_id: str, expected_path: str) -> bool:
    """Helper to check if an artifact exists at the given path."""
    dirname = os.path.dirname(expected_path)
    try:
        items = client.list_artifacts(run_id, dirname if dirname else None)
    except Exception:
        return False
    return any(item.path == expected_path or item.path.startswith(expected_path + "/") for item in items)


def _validate_telemetry_schema(kind: str, run_id: str):
    """Ensure the finished run strictly adheres to our telemetry contract."""
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    schema = TELEMETRY_SCHEMA[kind]

    # Parameters
    logged_params = set(run.data.params.keys())
    missing_params = set(schema["params"]["required"]) - logged_params
    if missing_params:
        raise TelemetryContractError(f"Run {run_id} (kind={kind}) validation failed: Missing required params: {missing_params}")

    # Tags
    logged_tags = set(run.data.tags.keys())
    missing_tags = set(schema["tags"]["required"]) - logged_tags
    if missing_tags:
        raise TelemetryContractError(f"Run {run_id} (kind={kind}) validation failed: Missing required tags: {missing_tags}")

    # Metrics
    logged_metrics = set(run.data.metrics.keys())
    missing_metrics = set(schema["metrics"]["required"]) - logged_metrics
    if missing_metrics:
        raise TelemetryContractError(f"Run {run_id} (kind={kind}) validation failed: Missing required metrics: {missing_metrics}")

    # Artifacts
    # Safely merge tags and params for formatting. Keys in params overwrite tags if they collide.
    fmt_context = {**run.data.tags, **run.data.params}

    for art_template in schema["artifacts"]["required"]:
        try:
            expected_path = art_template.format(**fmt_context)
        except KeyError as e:
            raise TelemetryContractError(f"Run {run_id} (kind={kind}): Cannot resolve artifact path '{art_template}' because param/tag {e} is missing in tracked data.")

        if not _check_artifact_exists(client, run_id, expected_path):
            raise TelemetryContractError(f"Run {run_id} (kind={kind}) validation failed: Missing required artifact: '{expected_path}'")


@contextmanager
def start_run(kind: str, run_name: str, tags: Mapping[str, Any] | None = None):
    _ensure_known_kind(kind)
    merged_tags = {"kind": kind, **(tags or {})}
    allowed_tags = set(TELEMETRY_SCHEMA[kind]["tags"]["required"] + TELEMETRY_SCHEMA[kind]["tags"]["optional"])
    _check_keys(kind, merged_tags, allowed_tags, "tag")

    start = datetime.now()
    print(f"[UPLOAD START] {start.strftime('%H:%M:%S')} — Starting MLflow run: {run_name} (kind={kind})")

    # The exception handling here allows MLflow's standard context manager to correctly 
    # catch anything that goes wrong during the block (including our contract validation)
    # and mark the run as FAILED automatically.
    with mlflow.start_run(run_name=run_name, tags=_clean_tags(merged_tags)) as run:
        yield run
        
        # When execution hits here, the user's `with` block completed without exceptions.
        # So we trigger the final telemetry contract validation!
        print(f"[VALIDATION] Enforcing telemetry contract for {run_name} (kind={kind})... ", end="")
        mlflow.flush_artifact_async_logging()
        _validate_telemetry_schema(kind, run.info.run_id)
        print("PASS")

    end = datetime.now()
    elapsed = (end - start).total_seconds()
    print(f"[UPLOAD  END ] {end.strftime('%H:%M:%S')} — Run {run_name} (kind={kind}) finished & validated in {elapsed:.1f}s")


def log_params(kind: str, params: Mapping[str, Any]) -> None:
    _ensure_known_kind(kind)
    cleaned = _clean_params(params)
    allowed_params = set(TELEMETRY_SCHEMA[kind]["params"]["required"] + TELEMETRY_SCHEMA[kind]["params"]["optional"])
    _check_keys(kind, cleaned, allowed_params, "param")
    if cleaned:
        with _timed_log(f"Logging params (kind={kind}, n={len(cleaned)})"):
            mlflow.log_params(cleaned)


def log_tags(kind: str, tags: Mapping[str, Any]) -> None:
    _ensure_known_kind(kind)
    cleaned = _clean_tags(tags)
    allowed_tags = set(TELEMETRY_SCHEMA[kind]["tags"]["required"] + TELEMETRY_SCHEMA[kind]["tags"]["optional"])
    _check_keys(kind, cleaned, allowed_tags, "tag")
    if cleaned:
        with _timed_log(f"Logging tags (kind={kind}, n={len(cleaned)})"):
            mlflow.set_tags(cleaned)
