#!/usr/bin/env python3
"""Backfill/overwrite `identity_hash` tags on existing model runs.

Use this after any change to the model identity hash schema (e.g. removing a
config field like dataset.num_classes from the identity hash). The script
recomputes each run's identity_hash using the current schema and the run's
stored config YAML artifact, then updates the MLflow tag.

By default the script is a dry-run. Pass --apply to write tags.

Usage:
    uv run scripts/migrate_model_identity_hashes.py --experiment ConTopo
    uv run scripts/migrate_model_identity_hashes.py --experiment ConTopo --apply
    uv run scripts/migrate_model_identity_hashes.py --experiment ConTopo --limit 5
"""
from __future__ import annotations

import argparse
import json
import logging
from typing import Dict

from mlflow.tracking import MlflowClient
import mlflow
from omegaconf import OmegaConf

from src.config.hash import identity_hash
from src.config.structured import DatasetConfig, LossConfig, ModelConfig, TrainingConfig


def _schema_keys(struct_class) -> dict:
    """Return the current-schema field template as a plain dict (all values are
    either nested dicts or None), used to filter stored configs."""
    return OmegaConf.to_container(OmegaConf.structured(struct_class))


def _filter_to_schema(stored: dict, template: dict) -> dict:
    """Recursively keep only keys that exist in the current schema template.

    This strips any field removed from the structured config (e.g. num_classes,
    split.seed) so that old stored configs hash the same as new configs.
    """
    result = {}
    for k, v in stored.items():
        if k not in template:
            continue
        if isinstance(v, dict) and isinstance(template[k], dict):
            result[k] = _filter_to_schema(v, template[k])
        else:
            result[k] = v
    return result


def _flatten_section(prefix: str, section: Dict) -> Dict[str, str]:
    """Flatten a config section to dot-path string fields.

    Must exactly mirror _flatten_identity_section in scripts/01_train_models.py.
    Lists are serialised with json.dumps (not str()) to match that function.
    """
    out: Dict[str, str] = {}

    def _walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                _walk(v, f"{path}.{k}")
            return
        if isinstance(node, list):
            out[path] = json.dumps(node, sort_keys=True)
            return
        out[path] = str(node)

    _walk(section, prefix)
    return out


def find_resolved_config_artifact(client: MlflowClient, run_id: str) -> str | None:
    """Return the artifact path of the stored config YAML, or None."""
    artifacts = client.list_artifacts(run_id, path="config")
    for art in artifacts:
        if art.path and art.path.endswith(".yaml"):
            return art.path
    return None


def load_resolved_cfg(run_id: str, artifact_relpath: str) -> dict | None:
    try:
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/{artifact_relpath}"
        )
        cfg = OmegaConf.load(local_path)
        return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        return None


def compute_model_identity_from_cfg(cfg: dict) -> str:
    """Compute the current-schema identity hash from a resolved config dict.

    Mirrors _model_identity_fields + identity_hash("model") in 01_train_models.py.
    Filters each section through the current structured config schema so that
    fields removed at any point (e.g. dataset.num_classes, dataset.split.seed)
    are excluded regardless of what the stored YAML contains.
    """
    schema_version = str(cfg.get("schema_version"))
    trial = str(cfg.get("trial"))
    seed = str(cfg.get("seed"))

    model_section = _filter_to_schema(cfg.get("model", {}), _schema_keys(ModelConfig))
    loss_section = _filter_to_schema(cfg.get("loss", {}), _schema_keys(LossConfig))
    dataset_section = _filter_to_schema(cfg.get("dataset", {}), _schema_keys(DatasetConfig))
    training_section = _filter_to_schema(cfg.get("training", {}), _schema_keys(TrainingConfig))

    fields: Dict[str, str] = {}
    fields.update(_flatten_section("model", model_section))
    fields.update(_flatten_section("loss", loss_section))
    fields.update(_flatten_section("dataset", dataset_section))
    fields.update(_flatten_section("training", training_section))

    return identity_hash(
        "model",
        schema_version=schema_version,
        trial=trial,
        seed=seed,
        **fields,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Backfill model identity_hash tags after schema changes."
    )
    parser.add_argument("--experiment", required=True, help="MLflow experiment name")
    parser.add_argument(
        "--apply", action="store_true", help="Actually write tags (default: dry-run)"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Cap number of runs processed (0=all)"
    )
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI")
    args = parser.parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    client = MlflowClient()
    exp = mlflow.get_experiment_by_name(args.experiment)
    if exp is None:
        raise SystemExit(f"Experiment not found: {args.experiment!r}")

    if not args.apply:
        print("DRY RUN — pass --apply to write tags.\n")

    filter_str = "tags.kind = 'model' and attributes.status = 'FINISHED'"
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id], filter_string=filter_str
    )
    print(f"Found {len(runs)} FINISHED model runs.")

    processed = 0
    updated = 0
    skipped = 0

    for _, row in runs.iterrows():
        run_id = row["run_id"]
        existing_identity = client.get_run(run_id).data.tags.get("identity_hash")

        artifact_rel = find_resolved_config_artifact(client, run_id)
        if artifact_rel is None:
            logging.warning("No config artifact for run %s; skipping", run_id)
            skipped += 1
            continue

        cfg = load_resolved_cfg(run_id, artifact_rel)
        if cfg is None:
            logging.warning("Failed loading config for run %s; skipping", run_id)
            skipped += 1
            continue

        try:
            new_identity = compute_model_identity_from_cfg(cfg)
        except Exception as e:
            logging.warning("Hash computation failed for run %s: %s; skipping", run_id, e)
            skipped += 1
            continue

        if existing_identity == new_identity:
            print(f"OK    {run_id}")
        else:
            print(
                f"PATCH {run_id}  {existing_identity or '(none)'} -> {new_identity}"
            )
            if args.apply:
                client.set_tag(run_id, "identity_hash", new_identity)
            updated += 1

        processed += 1
        if args.limit and processed >= args.limit:
            break

    print(f"\nDone. processed={processed}, patched={updated}, skipped={skipped}")
    if not args.apply and updated > 0:
        print("Re-run with --apply to write changes.")


if __name__ == "__main__":
    main()
