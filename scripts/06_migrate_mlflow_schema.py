#!/usr/bin/env python3
"""
Migrate MLflow params/tags placement to the current schema.

Default behavior is non-destructive:
  - copies values to the canonical destination (param vs tag)
  - keeps original source fields untouched

Optional cleanup can remove source-side duplicates after successful copy.

Examples:
        python scripts/06_migrate_mlflow_schema.py
        python scripts/06_migrate_mlflow_schema.py --kind model --apply
        python scripts/06_migrate_mlflow_schema.py --kind model --apply --remove-source
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import mlflow
from mlflow.entities import Run

from src.mlflow_schema_logger import ALLOWED_PARAMS, ALLOWED_TAGS


@dataclass
class MovePlan:
    run_id: str
    key: str
    source_slot: str  # "tag" | "param"
    destination_slot: str  # "tag" | "param"
    value: Any
    reason: str


@dataclass
class Conflict:
    run_id: str
    key: str
    source_slot: str
    destination_slot: str
    source_value: Any
    destination_value: Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate MLflow params/tags schema placement across runs."
    )
    parser.add_argument(
        "--tracking-uri",
        default="sqlite:///outputs/mlflow.db",
        help="MLflow tracking URI (default: sqlite:///outputs/mlflow.db)",
    )
    parser.add_argument(
        "--experiment-name",
        default="contopo",
        help="MLflow experiment name (default: contopo)",
    )
    parser.add_argument(
        "--kind",
        default="model",
        choices=sorted(ALLOWED_PARAMS.keys()),
        help="Run kind to migrate (default: model)",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Limit number of runs processed (0 = all)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes. Without this flag, runs in dry-run mode.",
    )
    parser.add_argument(
        "--remove-source",
        action="store_true",
        help=(
            "After copying value to canonical destination, remove value from source "
            "slot. Ignored in dry-run mode."
        ),
    )
    parser.add_argument(
        "--skip-finished-only",
        action="store_true",
        help="If set, process all run statuses; otherwise only FINISHED runs.",
    )
    return parser.parse_args()


def _coerce_param_value(value: Any) -> Any:
    if isinstance(value, (bool, int, float)):
        return value
    if value is None:
        return None

    s = str(value).strip()
    if s == "":
        return ""

    lower = s.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False

    try:
        if "." not in s and "e" not in lower:
            return int(s)
    except ValueError:
        pass

    try:
        return float(s)
    except ValueError:
        return s


def _build_filter(kind: str, finished_only: bool) -> str:
    base = f"tags.kind = '{kind}'"
    if finished_only:
        return f"{base} and attributes.status = 'FINISHED'"
    return base


def _collect_runs(
    experiment_id: str, kind: str, max_runs: int, finished_only: bool
) -> list[Run]:
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=_build_filter(kind=kind, finished_only=finished_only),
        output_format="list",
        max_results=max_runs if max_runs > 0 else 100000,
    )
    return runs


def _plan_run_migration(run: Run, kind: str) -> tuple[list[MovePlan], list[Conflict]]:
    allowed_params = ALLOWED_PARAMS[kind]
    allowed_tags = ALLOWED_TAGS[kind]

    params = dict(run.data.params)
    tags = dict(run.data.tags)

    plans: list[MovePlan] = []
    conflicts: list[Conflict] = []

    keys = set(params.keys()) | set(tags.keys())
    for key in keys:
        in_param = key in params
        in_tag = key in tags

        wants_param = key in allowed_params
        wants_tag = key in allowed_tags

        if wants_param and not wants_tag:
            if in_tag and not in_param:
                plans.append(
                    MovePlan(
                        run_id=run.info.run_id,
                        key=key,
                        source_slot="tag",
                        destination_slot="param",
                        value=tags[key],
                        reason="key_allowed_as_param_only",
                    )
                )
            elif in_tag and in_param and str(tags[key]) != str(params[key]):
                conflicts.append(
                    Conflict(
                        run_id=run.info.run_id,
                        key=key,
                        source_slot="tag",
                        destination_slot="param",
                        source_value=tags[key],
                        destination_value=params[key],
                    )
                )
        elif wants_tag and not wants_param:
            if in_param and not in_tag:
                plans.append(
                    MovePlan(
                        run_id=run.info.run_id,
                        key=key,
                        source_slot="param",
                        destination_slot="tag",
                        value=params[key],
                        reason="key_allowed_as_tag_only",
                    )
                )
            elif in_param and in_tag and str(params[key]) != str(tags[key]):
                conflicts.append(
                    Conflict(
                        run_id=run.info.run_id,
                        key=key,
                        source_slot="param",
                        destination_slot="tag",
                        source_value=params[key],
                        destination_value=tags[key],
                    )
                )

    return plans, conflicts


def _apply_plan(client: mlflow.tracking.MlflowClient, plan: MovePlan) -> None:
    if plan.destination_slot == "tag":
        client.set_tag(plan.run_id, plan.key, str(plan.value))
    elif plan.destination_slot == "param":
        v = _coerce_param_value(plan.value)
        client.log_param(plan.run_id, plan.key, v)
    else:
        raise ValueError(f"Unknown destination slot: {plan.destination_slot}")


def _remove_source(client: mlflow.tracking.MlflowClient, plan: MovePlan) -> None:
    if plan.source_slot == "tag":
        client.delete_tag(plan.run_id, plan.key)
        return

    # MLflow params are immutable: cannot delete in place.
    # We keep source params to remain non-destructive.


def main() -> None:
    args = _parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    experiment = mlflow.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        raise RuntimeError(
            f"Experiment '{args.experiment_name}' not found at '{args.tracking_uri}'."
        )

    runs = _collect_runs(
        experiment_id=experiment.experiment_id,
        kind=args.kind,
        max_runs=args.max_runs,
        finished_only=not args.skip_finished_only,
    )

    client = mlflow.tracking.MlflowClient()

    total_plans = 0
    total_conflicts = 0
    migrated_runs = 0

    print(
        f"Found {len(runs)} run(s) for kind='{args.kind}' in experiment='{args.experiment_name}'."
    )
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")

    for run in runs:
        plans, conflicts = _plan_run_migration(run, kind=args.kind)
        total_plans += len(plans)
        total_conflicts += len(conflicts)

        if not plans and not conflicts:
            continue

        migrated_runs += 1
        print(f"\nRun {run.info.run_id}:")

        for c in conflicts:
            print(
                "  [CONFLICT] "
                f"{c.key}: {c.source_slot}='{c.source_value}' vs "
                f"{c.destination_slot}='{c.destination_value}' (left unchanged)"
            )

        for plan in plans:
            print(
                f"  [MOVE] {plan.key}: {plan.source_slot} -> {plan.destination_slot} "
                f"(value='{plan.value}', reason={plan.reason})"
            )
            if args.apply:
                _apply_plan(client, plan)
                if args.remove_source:
                    _remove_source(client, plan)

    print("\nSummary")
    print(f"  Runs scanned: {len(runs)}")
    print(f"  Runs with actions/conflicts: {migrated_runs}")
    print(f"  Planned moves: {total_plans}")
    print(f"  Conflicts: {total_conflicts}")

    if args.apply and args.remove_source:
        print(
            "  Note: source tags can be removed, but source params are immutable in MLflow and are kept."
        )


if __name__ == "__main__":
    main()
