from __future__ import annotations

from typing import Any, Mapping

import mlflow

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
    return mlflow.start_run(run_name=run_name, tags=_clean_tags(merged_tags))


def log_params(kind: str, params: Mapping[str, Any]) -> None:
    _ensure_known_kind(kind)
    cleaned = _clean_params(params)
    _check_keys(kind, cleaned, ALLOWED_PARAMS[kind], "param")
    if cleaned:
        mlflow.log_params(cleaned)


def log_tags(kind: str, tags: Mapping[str, Any]) -> None:
    _ensure_known_kind(kind)
    cleaned = _clean_tags(tags)
    _check_keys(kind, cleaned, ALLOWED_TAGS[kind], "tag")
    if cleaned:
        mlflow.set_tags(cleaned)
