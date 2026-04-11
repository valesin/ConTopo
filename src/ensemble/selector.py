"""
Cfg-driven ensemble component selector.
Auto-discovers and groups models dynamically from MLflow.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any, Dict, List, Optional

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
    sample_size = cfg.groups.get("sample_size", None)
    return _discover(
        experiment_name=experiment_name,
        group_by=list(cfg.groups.group_by),
        min_components=int(cfg.groups.min_components),
        base_filter=dict(cfg.groups.filter) if cfg.groups.filter else {},
        sample_size=int(sample_size) if sample_size is not None else None,
    )


def _combo_hash(sorted_ids: List[str]) -> str:
    """6-char deterministic hash of a sorted run-id list."""
    canonical = json.dumps(sorted_ids, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:6]


def _discover(
    experiment_name: str,
    group_by: List[str],
    min_components: int,
    base_filter: Dict[str, Any],
    sample_size: Optional[int] = None,
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

    # 4. If sample_size is set, expand each group into k-combinations
    if sample_size is None:
        return final_ensembles

    if sample_size < 2:
        raise ValueError(f"sample_size must be >= 2, got {sample_size}")

    expanded: Dict[str, List[str]] = {}
    for g_name, r_ids in final_ensembles.items():
        if len(r_ids) < sample_size:
            continue
        for combo in itertools.combinations(r_ids, sample_size):
            combo_sorted = sorted(combo)
            short_hash = _combo_hash(combo_sorted)
            combo_name = f"{g_name}_k{sample_size}_{short_hash}"
            expanded[combo_name] = combo_sorted

    return expanded
