#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf

from src.config.hash import IDEMPOTENCY_REGISTRY, identity_hash


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
        raise ValueError(f"Experiment not found: {experiment_name}")

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect model identity inputs and recompute hash with fixed missing values."
    )
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment", required=True)
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
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    chosen_run_id = find_run_id(
        client, args.experiment, args.run_id, args.identity_hash_old
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
    required_exact = sorted([pattern for pattern in allowed_patterns if not pattern.endswith("*")])
    wildcard_groups = sorted([pattern for pattern in allowed_patterns if pattern.endswith("*")])

    missing_exact = [
        key
        for key in required_exact
        if (key not in all_identity_fields) or (all_identity_fields.get(key) in (None, "None"))
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
        "tracking_uri": args.tracking_uri,
        "experiment": args.experiment,
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

    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
