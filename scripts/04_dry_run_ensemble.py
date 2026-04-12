#!/usr/bin/env python3
"""
04_dry_run_ensemble.py — Dry-run overview of ensemble discovery.

Prints the component pool for each ensemble that would be processed by
04_run_ensemble.py, without loading any artifacts or writing to MLflow.

Usage:
    python scripts/04_dry_run_ensemble.py
    python scripts/04_dry_run_ensemble.py groups.sample_size=3
    python scripts/04_dry_run_ensemble.py groups.filter.topology=torus
"""

from __future__ import annotations

import itertools
import math

import hydra
import mlflow
from omegaconf import DictConfig

from src.ensemble.selector import discover_ensembles_from_cfg, _discover
from src.mlflow_utils import setup_mlflow, get_run_context


def _fetch_run_metadata(experiment_name: str, run_ids: list[str]) -> dict[str, dict]:
    """Return {run_id: {rho, trial, topology}} for each run_id."""
    client = mlflow.tracking.MlflowClient()
    meta = {}
    for rid in run_ids:
        try:
            r = client.get_run(rid)
            rho, trial, topology = get_run_context(r)
            meta[rid] = {"rho": rho, "trial": trial, "topology": topology}
        except Exception:
            meta[rid] = {"rho": "?", "trial": "?", "topology": "?"}
    return meta


def _combinations_count(n: int, k: int) -> int:
    return math.comb(n, k)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_mlflow(cfg)

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
    print(f"  sample_size: {sample_size if sample_size is not None else 'null  (full groups)'}")
    if base_filter:
        print(f"  filter     : {base_filter}")

    # Discover groups before applying sample_size expansion so we can show
    # the pool separately from the combinations it would generate.
    groups_before_expansion = _discover(
        experiment_name=cfg.mlflow.experiment_name,
        group_by=group_by,
        min_components=min_components,
        base_filter=base_filter,
        sample_size=None,          # always fetch the full pool first
    )

    if not groups_before_expansion:
        print("\n  No groups discovered — nothing to run.")
        return

    # Collect all unique run IDs across groups to fetch metadata in bulk.
    all_run_ids = sorted({rid for ids in groups_before_expansion.values() for rid in ids})
    meta = _fetch_run_metadata(cfg.mlflow.experiment_name, all_run_ids)

    total_ensembles = 0

    for group_name, run_ids in sorted(groups_before_expansion.items()):
        n = len(run_ids)
        combos = _combinations_count(n, sample_size) if sample_size is not None else 1
        total_ensembles += combos

        print(f"\n{'─'*60}")
        print(f"  group : {group_name}")
        print(f"  pool  : {n} components", end="")
        if sample_size is not None:
            print(f"  →  C({n},{sample_size}) = {combos} combinations")
        else:
            print(f"  →  1 ensemble (full pool)")

        print(f"  {'run_id':>12}   topology    rho       trial")
        print(f"  {'─'*12}   {'─'*10}  {'─'*8}  {'─'*5}")
        for rid in run_ids:
            m = meta.get(rid, {})
            print(f"  {rid[:12]}   {m.get('topology','?'):<10}  {m.get('rho','?'):<8}  {m.get('trial','?')}")

    print(f"\n{'='*60}")
    print(f"  Total ensemble runs that would be submitted: {total_ensembles}")
    if sample_size is not None:
        print(f"  (each combination × {len(list(cfg.ensemble.votes))} vote methods)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
