"""
Cfg-driven ensemble component selector.
Auto-discovers and groups models dynamically from MLflow.
"""

from __future__ import annotations

import hashlib
import itertools
import json

from mlflow.entities import Run
from omegaconf import DictConfig
from src.repositories.functional_run_repository import search_runs


def discover_ensembles_from_cfg(
    cfg: DictConfig,
    experiment_name: str,
) -> dict[str, list[str]]:
    """
    Auto-discovers ensemble groupings using discovery controls from cfg.groups.

    Returns:
       Dictionary of { 'ensemble_name': ['run_id_1', 'run_id_2', ...], ... }
    """
    sample_size = cfg.groups.get("sample_size", None)
    field_ranges_raw = cfg.groups.get("field_ranges", None)
    field_ranges = (
        {k: list(v) for k, v in field_ranges_raw.items()} if field_ranges_raw else None
    )
    groups, _ = _discover(
        experiment_name=experiment_name,
        group_by=list(cfg.groups.group_by),
        min_components=int(cfg.groups.min_components),
        base_filter=dict(cfg.groups.filter) if cfg.groups.filter else {},
        sample_size=int(sample_size) if sample_size is not None else None,
        field_ranges=field_ranges,
    )
    return groups


def discover_ensembles_with_runs_from_cfg(
    cfg: DictConfig,
    experiment_name: str,
) -> tuple[dict[str, list[str]], dict[str, Run]]:
    """
    Like discover_ensembles_from_cfg but also returns a run_index {run_id: Run}
    built from the same search_runs call, avoiding redundant per-run fetches.
    """
    sample_size = cfg.groups.get("sample_size", None)
    field_ranges_raw = cfg.groups.get("field_ranges", None)
    field_ranges = (
        {k: list(v) for k, v in field_ranges_raw.items()} if field_ranges_raw else None
    )
    return _discover(
        experiment_name=experiment_name,
        group_by=list(cfg.groups.group_by),
        min_components=int(cfg.groups.min_components),
        base_filter=dict(cfg.groups.filter) if cfg.groups.filter else {},
        sample_size=int(sample_size) if sample_size is not None else None,
        field_ranges=field_ranges,
    )


def _combo_hash(sorted_ids: list[str]) -> str:
    """6-char deterministic hash of a sorted run-id list."""
    canonical = json.dumps(sorted_ids, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:6]


def _passes_ranges(run: Run, field_ranges: dict[str, list[float]]) -> bool:
    for path, (lo, hi) in field_ranges.items():
        entity, _, field = path.partition(".")
        if entity == "tags":
            raw = run.data.tags.get(field)
        elif entity == "params":
            raw = run.data.params.get(field)
        else:
            continue
        if raw is None:
            return False
        try:
            val = float(raw)
        except (ValueError, TypeError):
            return False
        if not (lo <= val <= hi):
            return False
    return True


def _discover(
    experiment_name: str,
    group_by: list[str],
    min_components: int,
    base_filter: dict[str, object],
    sample_size: int | None = None,
    field_ranges: dict[str, list[float]] | None = None,
) -> tuple[dict[str, list[str]], dict[str, Run]]:
    # 1. Base MLflow fetch
    filter_string = "attributes.status = 'FINISHED' and tags.kind = 'model'"
    if base_filter:
        for k, v in base_filter.items():
            filter_string += f" and {k} = '{v}'"

    runs = search_runs(filter_string, output_format="list")

    if not runs:
        raise ValueError(f"No FINISHED models found matching: {filter_string}")

    # 1b. Numeric range filtering (post-fetch, covers tags and params)
    if field_ranges:
        runs = [r for r in runs if _passes_ranges(r, field_ranges)]

    if not runs:
        raise ValueError(f"No runs remain after applying field_ranges={field_ranges}")

    run_index: dict[str, Run] = {r.info.run_id: r for r in runs}

    # 2. Form groups based on distinct keys
    groups: dict[str, list[str]] = {}

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
    final_ensembles: dict[str, list[str]] = {}
    for g_name, r_ids in groups.items():
        if len(r_ids) >= min_components:
            final_ensembles[g_name] = sorted(r_ids)

    # 4. If sample_size is set, expand each group into k-combinations
    if sample_size is None:
        return final_ensembles, run_index

    if sample_size < 2:
        raise ValueError(f"sample_size must be >= 2, got {sample_size}")

    expanded: dict[str, list[str]] = {}
    for g_name, r_ids in final_ensembles.items():
        if len(r_ids) < sample_size:
            continue
        for combo in itertools.combinations(r_ids, sample_size):
            combo_sorted = sorted(combo)
            short_hash = _combo_hash(combo_sorted)
            combo_name = f"{g_name}_k{sample_size}_{short_hash}"
            expanded[combo_name] = combo_sorted

    return expanded, run_index
