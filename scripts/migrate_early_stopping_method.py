#!/usr/bin/env python3
"""Backfill the early_stopping_method param on existing model runs.

Before this migration, early stopping was hardcoded to val_acc in
01_train_models.py. The param was not logged to MLflow. This script
sets early_stopping_method=val_acc on every FINISHED model run that
is missing the param.

── USAGE ────────────────────────────────────────────────────────────────────
    # Preview changes (no writes):
    uv run scripts/migrate_early_stopping_method.py --experiment contopo

    # Apply:
    uv run scripts/migrate_early_stopping_method.py --experiment contopo --apply
"""

from __future__ import annotations

import argparse

import mlflow
from mlflow.tracking import MlflowClient
from src.repositories.functional_run_repository import (
    configure_run_repository,
    search_runs,
)

PARAM_KEY = "early_stopping_method"
BACKFILL_VALUE = "val_acc"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill early_stopping_method param on existing model runs."
    )
    parser.add_argument("--experiment", required=True, help="MLflow experiment name")
    parser.add_argument(
        "--apply", action="store_true", help="Actually write params (default: dry-run)"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Cap number of runs processed (0=all)"
    )
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI")
    args = parser.parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    client = MlflowClient()
    configure_run_repository(mlflow.get_tracking_uri(), args.experiment)

    if not args.apply:
        print("DRY RUN — pass --apply to write params.\n")

    filter_str = "tags.kind = 'model' and attributes.status = 'FINISHED'"
    runs = search_runs(filter_str, output_format="pandas")
    print(f"Found {len(runs)} FINISHED model runs.")

    processed = 0
    patched = 0
    skipped = 0

    for _, row in runs.iterrows():
        run_id = row["run_id"]
        existing = client.get_run(run_id).data.params.get(PARAM_KEY)

        if existing is not None:
            print(f"SKIP  {run_id}  ({PARAM_KEY}={existing!r} already set)")
            skipped += 1
        else:
            print(f"PATCH {run_id}  (set {PARAM_KEY}={BACKFILL_VALUE!r})")
            if args.apply:
                client.log_param(run_id, PARAM_KEY, BACKFILL_VALUE)
            patched += 1

        processed += 1
        if args.limit and processed >= args.limit:
            break

    print(f"\nDone. processed={processed}, patched={patched}, skipped={skipped}")
    if not args.apply and patched > 0:
        print("Re-run with --apply to write changes.")


if __name__ == "__main__":
    main()
