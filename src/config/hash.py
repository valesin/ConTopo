"""
cfg_hash — deterministic hash of experiment-semantic config only.

Excluded top-level keys (do NOT affect experiment results):
  runtime, mlflow, storage, hydra, pipeline

Included (experiment-semantic):
  schema_version, trial, seed, model.*, loss.*, dataset.*, training.*
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from omegaconf import DictConfig, OmegaConf

EXCLUDED_KEYS = frozenset(
    {
        "runtime",
        "mlflow",
        "storage",
        "hydra",
        "pipeline",
        "ensemble",
        "adapter",
        "migration",
    }
)


@dataclass(frozen=True)
class StepIdentity:
    identity_fields: tuple[str, ...]


IDEMPOTENCY_REGISTRY: dict[str, StepIdentity] = {
    "model": StepIdentity(
        identity_fields=(
            "schema_version",
            "trial",
            "seed",
            "model.*",
            "loss.*",
            "dataset.*",
            "training.*",
        )
    ),
    "inference": StepIdentity(identity_fields=("trained_model_run_id", "split")),
    "category_similarity_profile": StepIdentity(
        identity_fields=("parent_run_id", "anchor_spec_hash", "similarity_metric", "split")
    ),
    "diagnostics": StepIdentity(identity_fields=("parent_run_id", "diagnostic_metric")),
    "ensemble": StepIdentity(
        identity_fields=("component_set_hash", "split", "feature_type", "method")
    ),
    "diversity": StepIdentity(
        identity_fields=("component_set_hash", "diversity_metric", "split")
    ),
    "consistency": StepIdentity(
        identity_fields=("component_set_hash", "anchor_spec_hash", "split")
    ),
    "metalearner": StepIdentity(
        identity_fields=(
            "component_set_hash",
            "split",
            "feature_type",
            "anchor_spec",
            "meta_split_spec",
            "similarity_metric",
            "init_seed",
            "profile_mask",
            "meta_type",
        )
    ),
}


def _deep_sort(obj):
    """Recursively sort dicts by key for canonical serialisation."""
    if isinstance(obj, dict):
        return {k: _deep_sort(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_deep_sort(v) for v in obj]
    return obj


def cfg_hash(cfg: DictConfig) -> str:
    """
    Deterministic SHA-256 (16 hex chars) of experiment-semantic config.

    Steps:
      1. Resolve interpolations via OmegaConf.
      2. Strip EXCLUDED_KEYS from the top level.
      3. Canonicalise via ``json.dumps(sort_keys=True)``.
      4. Return first 16 hex chars of SHA-256.
    """
    resolved = OmegaConf.to_container(cfg, resolve=True)
    for key in EXCLUDED_KEYS:
        resolved.pop(key, None)
    canonical = json.dumps(_deep_sort(resolved), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _field_allowed(field_name: str, allowed_patterns: tuple[str, ...]) -> bool:
    """Return True if ``field_name`` matches one of the allowed exact/wildcard patterns."""
    for pattern in allowed_patterns:
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            if field_name.startswith(prefix):
                return True
        elif field_name == pattern:
            return True
    return False


def identity_hash(kind: str, **fields: Any) -> str:
    """Deterministic SHA-256 (16 hex chars) identity hash for a pipeline step."""
    if kind not in IDEMPOTENCY_REGISTRY:
        raise ValueError(f"Unknown idempotency kind: {kind}")

    allowed = IDEMPOTENCY_REGISTRY[kind].identity_fields
    provided = set(fields.keys())

    unknown = sorted(k for k in provided if not _field_allowed(k, allowed))
    if unknown:
        raise ValueError(
            f"Unknown identity field(s) for kind='{kind}': {unknown}. "
            f"Allowed patterns: {', '.join(allowed)}"
        )

    required_exact = {f for f in allowed if not f.endswith("*")}
    missing = sorted(required_exact - provided)
    if missing:
        raise ValueError(
            f"Missing identity field(s) for kind='{kind}': {missing}. "
            f"Required exact fields: {', '.join(sorted(required_exact))}"
        )

    wildcard_patterns = [f for f in allowed if f.endswith("*")]
    missing_wildcards = []
    for pattern in wildcard_patterns:
        prefix = pattern[:-1]
        if not any(k.startswith(prefix) for k in provided):
            missing_wildcards.append(pattern)
    if missing_wildcards:
        raise ValueError(
            f"Missing identity field groups for kind='{kind}': {', '.join(missing_wildcards)}. "
            "At least one field must match each wildcard group."
        )

    canonical = json.dumps(
        {"kind": kind, "fields": _deep_sort(fields)},
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def compute_anchor_spec_hash(
    source_split: str,
    per_class: int,
    strategy: str,
    order_by: str,
    num_classes: int,
) -> str:
    """Deterministic 16-char hex hash of an anchor specification."""
    spec_dict = {
        "source_split": source_split,
        "per_class": per_class,
        "strategy": strategy,
        "order_by": order_by,
        "num_classes": num_classes,
    }
    canonical = json.dumps(spec_dict, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def similarity_profile_hash(
    parent_run_id: str,
    anchor_spec_hash: str,
    similarity_metric: str,
    split: str = "test",
) -> str:
    """Compatibility wrapper for category similarity profile identity hashing."""
    return identity_hash(
        "category_similarity_profile",
        parent_run_id=parent_run_id,
        anchor_spec_hash=anchor_spec_hash,
        similarity_metric=similarity_metric,
        split=split,
    )


def component_set_hash(run_ids: list[str]) -> str:
    """Hash of sorted component model run_ids."""
    canonical = json.dumps(sorted(run_ids), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def behaviour_input_hash(
    component_set_hash_val: str,
    split: str = "test",
    feature_type: str = "logits",
    anchor_spec: str = "",
    meta_split_spec: str = "",
    similarity_metric: str = "",
    init_seed: str = "",
    profile_mask: str = "",
    method: str = "",
    meta_type: str = "",
) -> str:
    """Compatibility hash helper now including method/meta_type to avoid collisions."""
    parts = [
        component_set_hash_val,
        split,
        feature_type,
        anchor_spec,
        meta_split_spec,
        similarity_metric,
        init_seed,
        profile_mask,
        method,
        meta_type,
    ]
    canonical = json.dumps(parts, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def consistency_hash(cs_hash: str, anchor_spec_hash: str, split: str) -> str:
    """Compatibility wrapper for consistency-step identity hashing."""
    return identity_hash(
        "consistency",
        component_set_hash=cs_hash,
        anchor_spec_hash=anchor_spec_hash,
        split=split,
    )
