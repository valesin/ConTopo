"""
Declarative ensemble component selector (Tier-1).
Auto-discovers and groups models dynamically from MLflow.
"""

from __future__ import annotations
from typing import Any, Dict, List
import mlflow


def discover_ensembles(
    experiment_name: str,
    group_by: List[str] = ["topology", "rho"],
    min_components: int = 2,
    base_filter: Dict[str, Any] = None,
) -> Dict[str, List[str]]:
    """
    Auto-discovers ensemble groupings based on model params.

    Returns:
       Dictionary of { 'ensemble_name': ['run_id_1', 'run_id_2', ...], ... }
    """
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
        # Ensure model has all the necessary group_by tags
        if not all(k in params for k in group_by):
            continue

        # Create a unique group name based on the grouping keys
        # e.g., "topology_torus_rho_0.0"
        sig_parts = [f"{k}_{params[k]}" for k in group_by]
        group_name = "ens_" + "_".join(sig_parts)

        if group_name not in groups:
            groups[group_name] = []

        groups[group_name].append(r.info.run_id)

    # 3. Filter minimum component clusters and sort ID sequences perfectly
    final_ensembles = {}
    total_found = 0
    for g_name, r_ids in groups.items():
        if len(r_ids) >= min_components:
            # Sorted hashes guarantee the subset hash calculation is mathematically identical
            final_ensembles[g_name] = sorted(r_ids)
            total_found += 1

    return final_ensembles
