#!/usr/bin/env python3
"""Debug identity hash mismatches between stored MLflow runs and the training script.

Prints the stored hash, migration-computed hash, and full identity fields
for a given run, so you can compare against what 01_train_models.py computes.

Usage:
    uv run scripts/debug_identity_hash.py --trial 0 --rho 0.0
    uv run scripts/debug_identity_hash.py --trial 0 --rho 0.0 --tracking-uri sqlite:///outputs/mlflow.db
"""
from __future__ import annotations

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlflow
from mlflow.tracking import MlflowClient

from scripts.migrate_model_identity_hashes import (
    compute_model_identity_from_cfg,
    find_resolved_config_artifact,
    load_resolved_cfg,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", required=True, help="Trial index (e.g. 0)")
    parser.add_argument("--rho", required=True, help="Loss rho value (e.g. 0.0)")
    parser.add_argument("--experiment", default="contopo")
    parser.add_argument("--tracking-uri", default="sqlite:///outputs/mlflow.db")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    exp = mlflow.get_experiment_by_name(args.experiment)
    if exp is None:
        raise SystemExit(f"Experiment {args.experiment!r} not found")

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=(
            f"tags.kind = 'model' and "
            f"attributes.status = 'FINISHED' and "
            f"tags.trial = '{args.trial}' and "
            f"params.rho = '{args.rho}'"
        ),
        max_results=1,
        output_format="list",
    )
    if not runs:
        raise SystemExit(f"No FINISHED model run found for trial={args.trial} rho={args.rho}")

    r = runs[0]
    run_id = r.info.run_id
    stored_hash = r.data.tags.get("identity_hash", "(none)")

    artifact_rel = find_resolved_config_artifact(client, run_id)
    if artifact_rel is None:
        raise SystemExit(f"No config artifact found for run {run_id}")

    cfg_dict = load_resolved_cfg(run_id, artifact_rel)
    if cfg_dict is None:
        raise SystemExit(f"Failed to load config artifact for run {run_id}")

    migration_hash, fields = compute_model_identity_from_cfg(cfg_dict)

    print(f"run_id:         {run_id}")
    print(f"stored hash:    {stored_hash}")
    print(f"migration hash: {migration_hash}")
    print(f"match:          {stored_hash == migration_hash}")
    print(f"\nidentity fields:")
    print(json.dumps(dict(sorted(fields.items())), indent=2))


if __name__ == "__main__":
    main()
