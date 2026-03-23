"""
Cfg-driven ensemble component selector.
Auto-discovers and groups models dynamically from MLflow.
"""

from __future__ import annotations
from typing import Any, Dict, List

import mlflow
from omegaconf import DictConfig


def discover_ensembles_from_cfg(
    cfg: DictConfig,
    experiment_name: str,
) -> Dict[str, List[str]]:
    """
    Auto-discovers ensemble groupings using discovery controls from cfg.groups.

    Returns:
       Dictionary of { 'ensemble_name': ['run_id_1', 'run_id_2', ...], ... }
    """
    return _discover(
        experiment_name=experiment_name,
        group_by=list(cfg.groups.group_by),
        min_components=int(cfg.groups.min_components),
        base_filter=dict(cfg.groups.filter) if cfg.groups.filter else {},
    )


def _discover(
    experiment_name: str,
    group_by: List[str],
    min_components: int,
    base_filter: Dict[str, Any],
) -> Dict[str, List[str]]:
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"MLflow experiment '{experiment_name}' not found.")

    # 1. Base MLflow fetch
    filter_string = "attributes.status = 'FINISHED' and tags.kind = 'model'"
    if base_filter:
        for k, v in base_filter.items():
            filter_string += f" and params.{k} = '{v}'"

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=filter_string,
        output_format="list",
    )

    if not runs:
        raise ValueError(f"No FINISHED models found matching: {filter_string}")

    # 2. Form groups based on distinct keys
    groups: Dict[str, List[str]] = {}

    for r in runs:
        params = r.data.params
        if not all(k in params for k in group_by):
            continue

        sig_parts = [f"{k}_{params[k]}" for k in group_by]
        group_name = "ens_" + "_".join(sig_parts)

        if group_name not in groups:
            groups[group_name] = []

        groups[group_name].append(r.info.run_id)

    # 3. Filter minimum component clusters and sort ID sequences
    final_ensembles = {}
    for g_name, r_ids in groups.items():
        if len(r_ids) >= min_components:
            final_ensembles[g_name] = sorted(r_ids)

    return final_ensembles
