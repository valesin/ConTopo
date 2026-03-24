#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf

from src.config.hash import IDEMPOTENCY_REGISTRY, cfg_hash, identity_hash
from src.mlflow_utils import setup_mlflow


def flatten_training_style(prefix: str, section: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}

    def _walk(node: object, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                _walk(value, f"{path}.{key}")
            return
        if isinstance(node, list):
            out[path] = json.dumps(node, sort_keys=True)
            return
        out[path] = str(node)

    _walk(section, prefix)
    return out


def load_run_resolved_config(
    client: MlflowClient,
    run_id: str,
    artifact_uri: str | None,
    artifact_root_override: str | None,
) -> tuple[dict[str, Any], str]:
    yaml_path = None

    try:
        artifacts = client.list_artifacts(run_id, path="config")
        for art in artifacts:
            if art.path and art.path.endswith(".yaml"):
                yaml_path = art.path
                break
    except Exception:
        pass

    if yaml_path is None:
        try:
            root_arts = client.list_artifacts(run_id, path="")
            for art in root_arts:
                if art.path == "config" and art.is_dir:
                    nested = client.list_artifacts(run_id, path="config")
                    for nested_art in nested:
                        if nested_art.path and nested_art.path.endswith(".yaml"):
                            yaml_path = nested_art.path
                            break
                    break
        except Exception:
            pass

    if yaml_path is not None:
        local_file = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/{yaml_path}"
        )
        cfg = OmegaConf.load(local_file)
        resolved = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(resolved, dict):
            raise ValueError("Resolved config artifact is not a dictionary.")
        return resolved, yaml_path

    candidate_roots: list[Path] = []
    if artifact_root_override:
        candidate_roots.append(Path(artifact_root_override))
    if artifact_uri:
        uri_path = (
            Path(artifact_uri.replace("file://", "", 1))
            if artifact_uri.startswith("file://")
            else Path(artifact_uri)
        )
        candidate_roots.append(uri_path)
        if uri_path.name == "artifacts":
            candidate_roots.append(uri_path.parent.parent)

    for root in candidate_roots:
        if not root.exists():
            continue
        direct = root / "config"
        if direct.exists() and direct.is_dir():
            yamls = sorted(direct.glob("*.yaml"))
            if yamls:
                cfg = OmegaConf.load(str(yamls[0]))
                resolved = OmegaConf.to_container(cfg, resolve=True)
                if isinstance(resolved, dict):
                    return resolved, str(yamls[0])

        run_artifacts = root / run_id / "artifacts" / "config"
        if run_artifacts.exists() and run_artifacts.is_dir():
            yamls = sorted(run_artifacts.glob("*.yaml"))
            if yamls:
                cfg = OmegaConf.load(str(yamls[0]))
                resolved = OmegaConf.to_container(cfg, resolve=True)
                if isinstance(resolved, dict):
                    return resolved, str(yamls[0])

    raise FileNotFoundError(
        f"No resolved config YAML could be found for run {run_id}. "
        "Checked MLflow artifact API and filesystem fallbacks."
    )


def build_identity_input(cfg: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    fields = {
        "schema_version": str(cfg.get("schema_version")),
        "trial": str(cfg.get("trial")),
        "seed": str(cfg.get("seed")),
    }

    flattened: dict[str, str] = {}
    flattened.update(flatten_training_style("model", cfg.get("model", {}) or {}))
    flattened.update(flatten_training_style("loss", cfg.get("loss", {}) or {}))
    flattened.update(flatten_training_style("dataset", cfg.get("dataset", {}) or {}))
    flattened.update(flatten_training_style("training", cfg.get("training", {}) or {}))
    return fields, flattened


def dot_get(cfg: dict[str, Any], key: str) -> Any:
    node: Any = cfg
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def parse_overrides(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError(
            "--fixed-json must be a JSON object mapping field_key -> value"
        )
    return {str(k): str(v) for k, v in obj.items()}


def parse_hydra_overrides(raw: str | None) -> list[str]:
    if raw is None:
        return []
    values = [v.strip() for v in raw.split(",") if v.strip()]
    return values


def resolve_seed_local(cfg: Any) -> int:
    if cfg.seed is not None:
        return int(cfg.seed)
    return 100 + int(cfg.trial)


def compute_model_identity_from_cfg_dict(
    cfg_dict: dict[str, Any],
) -> tuple[str, dict[str, str]]:
    base_fields, flattened = build_identity_input(cfg_dict)
    all_fields = {**base_fields, **flattened}
    computed = identity_hash("model", **all_fields)
    return computed, all_fields


def compare_with_current_config(
    current_overrides: list[str],
    old_all_identity_fields: dict[str, str],
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parent.parent
    conf_dir = repo_root / "conf"

    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        current_cfg = compose(config_name="config", overrides=current_overrides)

    current_seed = resolve_seed_local(current_cfg)
    current_cfg.seed = current_seed

    current_cfg_dict = OmegaConf.to_container(current_cfg, resolve=True)
    if not isinstance(current_cfg_dict, dict):
        raise ValueError("Current composed config did not resolve to a dictionary")

    current_identity, current_all_identity_fields = (
        compute_model_identity_from_cfg_dict(current_cfg_dict)
    )
    current_cfg_hash = cfg_hash(current_cfg)

    all_keys = sorted(
        set(old_all_identity_fields.keys()) | set(current_all_identity_fields.keys())
    )
    changed: list[dict[str, str]] = []
    only_old: list[dict[str, str]] = []
    only_current: list[dict[str, str]] = []

    for key in all_keys:
        old_val = old_all_identity_fields.get(key)
        cur_val = current_all_identity_fields.get(key)
        if old_val is None and cur_val is not None:
            only_current.append({"key": key, "current": cur_val})
        elif cur_val is None and old_val is not None:
            only_old.append({"key": key, "old": old_val})
        elif old_val != cur_val:
            changed.append({"key": key, "old": old_val, "current": cur_val})

    return {
        "overrides": current_overrides,
        "current_cfg_hash": current_cfg_hash,
        "current_computed_identity_hash": current_identity,
        "old_vs_current": {
            "num_changed": len(changed),
            "num_only_old": len(only_old),
            "num_only_current": len(only_current),
            "changed_fields": changed,
            "only_old_fields": only_old,
            "only_current_fields": only_current,
        },
    }


def find_run_id(
    client: MlflowClient,
    experiment_name: str,
    run_id: str | None,
    identity_hash_old: str | None,
) -> str:
    if run_id:
        return run_id
    if not identity_hash_old:
        raise ValueError("Provide either --run-id or --identity-hash-old")

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        available = [
            e.name
            for e in client.search_experiments(
                view_type=mlflow.entities.ViewType.ACTIVE_ONLY
            )
        ]
        raise ValueError(
            f"Experiment not found: {experiment_name}. "
            f"Available active experiments: {available}"
        )

    rows = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=(
            "tags.kind = 'model' and "
            "attributes.status = 'FINISHED' and "
            f"tags.identity_hash = '{identity_hash_old}'"
        ),
    )
    if rows.empty:
        raise ValueError(
            f"No FINISHED model run found with tags.identity_hash={identity_hash_old} in experiment={experiment_name}"
        )
    if len(rows) > 1:
        raise ValueError(
            f"Multiple runs ({len(rows)}) matched identity_hash={identity_hash_old}; pass --run-id explicitly"
        )
    return str(rows.iloc[0]["run_id"])


def initialize_mlflow_like_pipeline(
    tracking_uri: str | None,
    experiment_name: str,
) -> tuple[str, str]:
    repo_root = Path(__file__).resolve().parent.parent
    conf_dir = repo_root / "conf"

    overrides = [f"mlflow.experiment_name={experiment_name}"]
    if tracking_uri:
        overrides.append(f"mlflow.tracking_uri={tracking_uri}")

    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    setup_mlflow(cfg)
    return mlflow.get_tracking_uri(), str(cfg.mlflow.experiment_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect model identity inputs and recompute hash with fixed missing values."
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help=(
            "MLflow tracking URI. Optional on server: if omitted, uses MLFLOW_TRACKING_URI "
            "from environment when available, otherwise MLflow default URI."
        ),
    )
    parser.add_argument(
        "--experiment",
        default="contopo",
        help="MLflow experiment name (default: contopo).",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--identity-hash-old", default=None)
    parser.add_argument(
        "--fixed-json",
        default=None,
        help='JSON object of field overrides, e.g. {"training.save_freq_epochs": "1"}',
    )
    parser.add_argument(
        "--allow-override-existing",
        action="store_true",
        help="Allow --fixed-json to override existing fields (default: only fill missing fields).",
    )
    parser.add_argument(
        "--artifact-root-override",
        default=None,
        help="Optional local path fallback root to locate run artifacts when MLflow artifact API cannot resolve them.",
    )
    parser.add_argument(
        "--compare-current-overrides",
        default=None,
        help=(
            "Comma-separated Hydra overrides for current config comparison, "
            "e.g. 'loss.rho=0.008,trial=1'."
        ),
    )
    args = parser.parse_args()

    env_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    chosen_tracking_uri = args.tracking_uri or env_tracking_uri
    active_tracking_uri, effective_experiment = initialize_mlflow_like_pipeline(
        tracking_uri=chosen_tracking_uri,
        experiment_name=args.experiment,
    )

    client = MlflowClient()
    chosen_run_id = find_run_id(
        client, effective_experiment, args.run_id, args.identity_hash_old
    )

    run = client.get_run(chosen_run_id)
    cfg, cfg_artifact_path = load_run_resolved_config(
        client=client,
        run_id=chosen_run_id,
        artifact_uri=run.info.artifact_uri,
        artifact_root_override=args.artifact_root_override,
    )

    base_fields, flattened = build_identity_input(cfg)
    all_identity_fields = {**base_fields, **flattened}

    allowed_patterns = IDEMPOTENCY_REGISTRY["model"].identity_fields
    required_exact = sorted(
        [pattern for pattern in allowed_patterns if not pattern.endswith("*")]
    )
    wildcard_groups = sorted(
        [pattern for pattern in allowed_patterns if pattern.endswith("*")]
    )

    missing_exact = [
        key
        for key in required_exact
        if (key not in all_identity_fields)
        or (all_identity_fields.get(key) in (None, "None"))
    ]

    wildcard_presence = {
        pattern: any(key.startswith(pattern[:-1]) for key in flattened.keys())
        for pattern in wildcard_groups
    }

    missing_wildcard_groups = [
        k for k, present in wildcard_presence.items() if not present
    ]

    fixed = parse_overrides(args.fixed_json)

    missing_dot_keys = sorted(k for k in fixed.keys() if k not in all_identity_fields)
    ignored_existing_dot_keys: list[str] = []

    if args.allow_override_existing:
        recompute_fields = dict(all_identity_fields)
        recompute_fields.update(fixed)
    else:
        recompute_fields = dict(all_identity_fields)
        for key, value in fixed.items():
            if key in recompute_fields:
                ignored_existing_dot_keys.append(key)
            else:
                recompute_fields[key] = value

    computed_from_run = identity_hash("model", **all_identity_fields)
    recomputed_with_fixed = identity_hash("model", **recompute_fields)

    output = {
        "tracking_uri": active_tracking_uri,
        "experiment": effective_experiment,
        "run_id": chosen_run_id,
        "existing_identity_hash_tag": run.data.tags.get("identity_hash"),
        "cfg_hash_tag": run.data.tags.get("cfg_hash"),
        "config_artifact": cfg_artifact_path,
        "required_exact_fields": required_exact,
        "allowed_identity_patterns": list(allowed_patterns),
        "missing_exact_fields": missing_exact,
        "wildcard_group_presence": wildcard_presence,
        "missing_wildcard_groups": missing_wildcard_groups,
        "flattened_identity_field_count": len(flattened),
        "flattened_identity_fields": flattened,
        "computed_identity_from_run_fields": computed_from_run,
        "fixed_values_input": fixed,
        "missing_keys_in_run_for_fixed_values": missing_dot_keys,
        "ignored_fixed_keys_already_present": sorted(ignored_existing_dot_keys),
        "effective_recompute_field_count": len(recompute_fields),
        "recomputed_identity_with_fixed_values": recomputed_with_fixed,
        "differs_after_fix": recomputed_with_fixed != computed_from_run,
    }

    compare_overrides = parse_hydra_overrides(args.compare_current_overrides)
    if compare_overrides:
        output["current_config_comparison"] = compare_with_current_config(
            current_overrides=compare_overrides,
            old_all_identity_fields=all_identity_fields,
        )

    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
