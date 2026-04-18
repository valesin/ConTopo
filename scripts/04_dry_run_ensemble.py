#!/usr/bin/env python3
"""
04_dry_run_ensemble.py — Dry-run overview of ensemble discovery.

Prints the component pool for each ensemble that would be processed by
04_run_ensemble.py, without loading any artifacts or writing to MLflow.

For each group, only fields with ≥ 2 distinct values are shown as columns,
making unexpected variation (forgotten group_by keys) immediately visible.

Usage:
    python scripts/04_dry_run_ensemble.py
    python scripts/04_dry_run_ensemble.py groups.sample_size=3
    python scripts/04_dry_run_ensemble.py "groups.filter={params.topology: torus}"
"""

from __future__ import annotations

import math

import hydra
from omegaconf import DictConfig

from src.ensemble.selector import _discover
from src.mlflow_utils import setup_mlflow
from src.repositories.functional_run_repository import (
    configure_run_repository,
    get_run,
)

_EXCLUDE_FIELDS = {"kind", "identity_hash", "cfg_hash", "run_name"}


def _fetch_run_metadata(run_ids: list[str]) -> dict[str, dict]:
    """Return {run_id: {field: value, ...}} with all params+tags, excluding noise."""
    meta = {}
    for rid in run_ids:
        try:
            r = get_run(rid)
            fields = {}
            for k, v in r.data.params.items():
                if not k.startswith("mlflow.") and k not in _EXCLUDE_FIELDS:
                    fields[f"param.{k}"] = v
            for k, v in r.data.tags.items():
                if not k.startswith("mlflow.") and k not in _EXCLUDE_FIELDS:
                    fields[f"tag.{k}"] = v
            meta[rid] = fields
        except Exception:
            meta[rid] = {}
    return meta


_ABSENT = "(absent)"


def _varying_fields(run_ids: list[str], meta: dict[str, dict]) -> list[str]:
    """Return sorted list of field names with >= 2 distinct values in this group."""
    all_keys = {k for rid in run_ids for k in meta.get(rid, {})}
    varying = []
    for k in sorted(all_keys):
        values = {meta.get(rid, {}).get(k) for rid in run_ids}
        if len(values) >= 2:
            varying.append(k)
    return varying


def _combinations_count(n: int, k: int) -> int:
    return math.comb(n, k)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_mlflow(cfg)
    configure_run_repository(cfg.mlflow.tracking_uri, cfg.mlflow.experiment_name)

    group_by = list(cfg.groups.group_by)
    min_components = int(cfg.groups.min_components)
    sample_size = cfg.groups.get("sample_size", None)
    if sample_size is not None:
        sample_size = int(sample_size)
    base_filter = dict(cfg.groups.filter) if cfg.groups.filter else {}

    print(f"\n{'='*60}")
    print("Ensemble Discovery — Dry Run")
    print(f"{'='*60}")
    print(f"  experiment : {cfg.mlflow.experiment_name}")
    print(f"  group_by   : {group_by}")
    print(f"  min_comps  : {min_components}")
    print(
        f"  sample_size: {sample_size if sample_size is not None else 'null  (full groups)'}"
    )
    if base_filter:
        print(f"  filter     : {base_filter}")

    # Discover groups before applying sample_size expansion so we can show
    # the pool separately from the combinations it would generate.
    groups_before_expansion = _discover(
        experiment_name=cfg.mlflow.experiment_name,
        group_by=group_by,
        min_components=min_components,
        base_filter=base_filter,
        sample_size=None,  # always fetch the full pool first
    )

    if not groups_before_expansion:
        print("\n  No groups discovered — nothing to run.")
        return

    # Collect all unique run IDs across groups to fetch metadata in bulk.
    all_run_ids = sorted(
        {rid for ids in groups_before_expansion.values() for rid in ids}
    )
    meta = _fetch_run_metadata(all_run_ids)

    total_ensembles = 0

    for group_name, run_ids in sorted(groups_before_expansion.items()):
        n = len(run_ids)
        combos = _combinations_count(n, sample_size) if sample_size is not None else 1
        total_ensembles += combos

        varying = _varying_fields(run_ids, meta)

        print(f"\n{'─'*60}")
        print(f"  group : {group_name}")
        print(f"  pool  : {n} components", end="")
        if sample_size is not None:
            print(f"  →  C({n},{sample_size}) = {combos} combinations")
        else:
            print(f"  →  1 ensemble (full pool)")

        if not varying:
            print("  varying fields: (none)")
            continue

        print(f"  varying fields: {', '.join(varying)}")

        # Build dynamic column widths.
        col_widths = {
            f: max(
                len(f),
                max(
                    (len(str(meta.get(rid, {}).get(f, _ABSENT))) for rid in run_ids),
                    default=1,
                ),
            )
            for f in varying
        }
        run_id_w = 12

        header = f"  {'run_id':>{run_id_w}}"
        divider = f"  {'─'*run_id_w}"
        for f in varying:
            w = col_widths[f]
            header += f"   {f:<{w}}"
            divider += f"   {'─'*w}"

        print(header)
        print(divider)
        for rid in run_ids:
            row = f"  {rid[:run_id_w]}"
            for f in varying:
                w = col_widths[f]
                val = str(meta.get(rid, {}).get(f, _ABSENT))
                row += f"   {val:<{w}}"
            print(row)

    print(f"\n{'='*60}")
    print(f"  Total ensemble runs that would be submitted: {total_ensembles}")
    if sample_size is not None:
        print(f"  (each combination × {len(list(cfg.ensemble.votes))} vote methods)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
