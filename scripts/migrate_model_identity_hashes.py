#!/usr/bin/env python3
"""Backfill/overwrite `identity_hash` tags for legacy model runs.

Usage:
    python scripts/migrate_model_identity_hashes.py --experiment my_experiment [--apply]

By default the script runs as a dry-run and prints proposed changes. Use
`--apply` to actually set the tag on MLflow runs.
"""
from __future__ import annotations

import argparse
import logging
from typing import Dict

from mlflow.tracking import MlflowClient
import mlflow
from omegaconf import OmegaConf

from src.config.hash import identity_hash


def _flatten_section(prefix: str, section: Dict) -> Dict[str, str]:
    out: Dict[str, str] = {}

    def _walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                _walk(v, f"{path}.{k}")
            return
        if isinstance(node, (list, tuple)):
            out[path] = str(node)
            return
        out[path] = str(node)

    _walk(section, prefix)
    return out


def find_resolved_config_artifact(client: MlflowClient, run_id: str) -> str | None:
    # Look for an artifact under path 'config' and return its path if found
    artifacts = client.list_artifacts(run_id, path="config")
    for art in artifacts:
        if art.path and art.path.endswith(".yaml"):
            return f"config/{art.path}" if art.path != "" else "config"
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
    # Mirror logic from scripts/01_train_models.py
    schema_version = str(cfg.get("schema_version"))
    trial = str(cfg.get("trial"))
    seed = str(cfg.get("seed"))

    model_section = cfg.get("model", {})
    loss_section = cfg.get("loss", {})
    dataset_section = cfg.get("dataset", {})
    training_section = cfg.get("training", {})

    fields = {}
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="MLflow experiment name")
    parser.add_argument(
        "--apply", action="store_true", help="Actually write tags to MLflow"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit number of runs to process (0=all)"
    )
    args = parser.parse_args()

    client = MlflowClient()
    exp = mlflow.get_experiment_by_name(args.experiment)
    if exp is None:
        raise SystemExit(f"Experiment not found: {args.experiment}")

    filter_str = "tags.kind = 'model' and attributes.status = 'FINISHED'"
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id], filter_string=filter_str
    )

    processed = 0
    updated = 0
    for _, row in runs.iterrows():
        run_id = row["run_id"]
        tags = client.get_run(run_id).data.tags
        existing_identity = tags.get("identity_hash")

        artifact_rel = find_resolved_config_artifact(client, run_id)
        if artifact_rel is None:
            logging.warning("No resolved config artifact for run %s; skipping", run_id)
            continue

        cfg = load_resolved_cfg(run_id, artifact_rel)
        if cfg is None:
            logging.warning(
                "Failed loading config artifact for run %s; skipping", run_id
            )
            continue

        new_identity = compute_model_identity_from_cfg(cfg)

        if existing_identity == new_identity:
            print(f"OK    {run_id} identity matches")
        else:
            print(f"PATCH {run_id} existing={existing_identity} -> new={new_identity}")
            if args.apply:
                client.set_tag(run_id, "identity_hash", new_identity)
                updated += 1

        processed += 1
        if args.limit and processed >= args.limit:
            break

    print(f"Processed: {processed}, Updated: {updated}")


if __name__ == "__main__":
    main()
