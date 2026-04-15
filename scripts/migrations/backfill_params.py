#!/usr/bin/env python3
"""Generic param backfill for FINISHED model runs.

Reads a migration spec (YAML) that declares which MLflow params to add and what
migration default each one should take, then writes the missing params to every
FINISHED model run in the target experiment.

The script is idempotent: if a run already has a param set it is left unchanged.

── SPEC FORMAT ──────────────────────────────────────────────────────────────
    # scripts/migrations/specs/<name>.yaml
    description: "Short human-readable description of this migration"
    params:
      param_name: "string_default_value"   # MLflow stores all params as strings
      another_param: "None"                 # conditional fields that were not used
      yet_another: "False"

Values are always stored as strings (MLflow param storage format).  Use Python
literal strings: "None", "False", "0.0", etc.

── USAGE ────────────────────────────────────────────────────────────────────
    # Preview (no writes):
    uv run scripts/migrations/backfill_params.py \\
        --spec scripts/migrations/specs/<name>.yaml \\
        --experiment <experiment_name>

    # Apply:
    uv run scripts/migrations/backfill_params.py \\
        --spec scripts/migrations/specs/<name>.yaml \\
        --experiment <experiment_name> --apply

    # Limit to first N runs (spot-check):
    uv run scripts/migrations/backfill_params.py \\
        --spec scripts/migrations/specs/<name>.yaml \\
        --experiment <experiment_name> --limit 5

    # Custom tracking URI:
    uv run scripts/migrations/backfill_params.py \\
        --spec scripts/migrations/specs/<name>.yaml \\
        --experiment <experiment_name> \\
        --tracking-uri sqlite:///outputs/mlflow.db

── RUNNING ORDER ────────────────────────────────────────────────────────────
    For hash-included param changes: run this script first, then run
    scripts/migrations/rehash_identities.py to update identity_hash tags.
    See CONTRIBUTING_AND_UPDATING.md §11 for the full protocol.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient

from src.repositories.functional_run_repository import (
    configure_run_repository,
    search_runs,
)


def load_spec(spec_path: str) -> tuple[str, dict[str, str]]:
    """Load and validate a migration spec YAML.

    Returns (description, params_dict).
    """
    with open(spec_path) as f:
        spec = yaml.safe_load(f)

    if not isinstance(spec, dict):
        raise ValueError(f"Spec file must be a YAML mapping, got {type(spec)}")
    if "params" not in spec:
        raise ValueError("Spec file must have a 'params' key")
    if not isinstance(spec["params"], dict):
        raise ValueError("'params' must be a YAML mapping of param_name: default_value")

    description = spec.get("description", Path(spec_path).stem)
    params = {str(k): str(v) for k, v in spec["params"].items()}
    return description, params


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill missing MLflow params on FINISHED model runs from a spec file."
    )
    parser.add_argument(
        "--spec",
        required=True,
        help="Path to the migration spec YAML (e.g. scripts/migrations/specs/my_migration.yaml)",
    )
    parser.add_argument("--experiment", required=True, help="MLflow experiment name")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write params (default: dry-run)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap number of runs processed (0 = all)",
    )
    parser.add_argument(
        "--tracking-uri",
        default="sqlite:///outputs/mlflow.db",
        help="MLflow tracking URI (default: sqlite:///outputs/mlflow.db)",
    )
    args = parser.parse_args()

    description, backfill_params = load_spec(args.spec)

    print(f"Migration: {description}")
    print(f"Params:    {list(backfill_params.keys())}")
    print()

    mlflow.set_tracking_uri(args.tracking_uri)
    configure_run_repository(args.tracking_uri, args.experiment)
    client = MlflowClient()

    if not args.apply:
        print("DRY RUN — pass --apply to write params.\n")

    filter_str = "tags.kind = 'model' and attributes.status = 'FINISHED'"
    runs = search_runs(filter_str, output_format="pandas")
    print(f"Found {len(runs)} FINISHED model runs.")

    processed = 0
    total_patched = 0
    total_skipped = 0

    for _, row in runs.iterrows():
        run_id = row["run_id"]
        run_data = client.get_run(run_id).data.params

        patched_this_run: list[str] = []
        skipped_this_run: list[str] = []

        for key, default_value in backfill_params.items():
            if key in run_data:
                skipped_this_run.append(key)
            else:
                patched_this_run.append(key)
                if args.apply:
                    client.log_param(run_id, key, default_value)

        if patched_this_run:
            status = "PATCH" if args.apply else "WOULD PATCH"
            print(
                f"{status} {run_id}  "
                f"(adding: {', '.join(f'{k}={backfill_params[k]}' for k in patched_this_run)})"
            )
            total_patched += len(patched_this_run)
        else:
            print(f"SKIP  {run_id}  (all params already present)")

        total_skipped += len(skipped_this_run)
        processed += 1
        if args.limit and processed >= args.limit:
            break

    print(
        f"\nDone. runs_processed={processed}, "
        f"params_patched={total_patched}, params_skipped={total_skipped}"
    )
    if not args.apply and total_patched > 0:
        print("Re-run with --apply to write changes.")


if __name__ == "__main__":
    main()
