#!/usr/bin/env python3
"""Soft-delete all non-model MLflow runs in a given experiment.

Deletes runs whose `kind` tag is anything other than "model". Model runs are
left untouched. Deletion is soft: runs are marked DELETED in the tracking
server but artifacts remain on S3 until `mlflow gc` is run separately.

── USAGE ────────────────────────────────────────────────────────────────────
    # Dry run — prints counts per kind, no changes:
    uv run scripts/migrations/delete_non_model_runs.py

    # Dry run against a specific experiment:
    uv run scripts/migrations/delete_non_model_runs.py --experiment contopo2

    # Execute deletions:
    uv run scripts/migrations/delete_non_model_runs.py --execute

    # Custom tracking URI:
    uv run scripts/migrations/delete_non_model_runs.py \\
        --tracking-uri https://dagshub.com/valesin/ConTopoTEST.mlflow --execute

The script reads MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_NAME from the
environment when --tracking-uri / --experiment are not provided explicitly.
Source .env.secrets before running to point at the production server.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import mlflow
from mlflow.tracking import MlflowClient

KEEP_KIND = "model"
PAGE_SIZE = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "contopo"),
        help="MLflow experiment name (default: $MLFLOW_EXPERIMENT_NAME or 'contopo')",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI"),
        help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete runs. Without this flag the script is a dry run.",
    )
    return parser.parse_args()


def fetch_all_runs(client: MlflowClient, experiment_id: str) -> list:
    """Page through all ACTIVE runs in the experiment."""
    runs = []
    token = None
    while True:
        page = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="attributes.status != 'DELETED'",
            max_results=PAGE_SIZE,
            page_token=token,
        )
        runs.extend(page)
        token = page.token
        if not token:
            break
    return runs


def main() -> None:
    args = parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    client = MlflowClient()

    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        print(f"ERROR: experiment '{args.experiment}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Experiment : {args.experiment}  (id={experiment.experiment_id})")
    print(f"Tracking   : {mlflow.get_tracking_uri()}")
    print(
        f"Mode       : {'EXECUTE — runs will be deleted' if args.execute else 'DRY RUN — no changes'}"
    )
    print()

    print("Fetching all runs…")
    all_runs = fetch_all_runs(client, experiment.experiment_id)
    print(f"  Total active runs found: {len(all_runs)}")
    print()

    # Partition by kind
    to_delete: list = []
    kind_counts: Counter = Counter()

    for run in all_runs:
        kind = run.data.tags.get("kind", "<no kind>")
        kind_counts[kind] += 1
        if kind != KEEP_KIND:
            to_delete.append(run)

    # Print summary table
    print("Run counts by kind:")
    for kind, count in sorted(kind_counts.items()):
        marker = "  KEEP" if kind == KEEP_KIND else "  DELETE"
        print(f"  {kind:<35} {count:>6}  {marker}")
    print()
    print(f"  Runs to delete : {len(to_delete)}")
    print(f"  Runs to keep   : {kind_counts.get(KEEP_KIND, 0)}")
    print()

    if not to_delete:
        print("Nothing to delete. Exiting.")
        return

    if not args.execute:
        print("DRY RUN complete. Re-run with --execute to apply deletions.")
        return

    # Confirm before proceeding
    answer = input(
        f"About to soft-delete {len(to_delete)} runs. Type 'yes' to confirm: "
    )
    if answer.strip().lower() != "yes":
        print("Aborted.")
        return

    print(f"\nDeleting {len(to_delete)} runs…")
    errors = 0
    for i, run in enumerate(to_delete, 1):
        try:
            client.delete_run(run.info.run_id)
            if i % 100 == 0 or i == len(to_delete):
                print(f"  {i}/{len(to_delete)} deleted")
        except Exception as e:
            print(f"  ERROR deleting run {run.info.run_id}: {e}", file=sys.stderr)
            errors += 1

    print()
    if errors:
        print(f"Done with {errors} error(s). Check stderr output above.")
    else:
        print("Done. All runs soft-deleted.")
    print()
    print("Artifacts are still on S3. To permanently free storage, run mlflow gc.")
    print("See the instructions printed when you run this script with --help, or")
    print("refer to the gc section in the project README.")


if __name__ == "__main__":
    main()
