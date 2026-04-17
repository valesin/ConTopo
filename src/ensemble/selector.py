"""
Cfg-driven ensemble component selector.
Auto-discovers and groups models dynamically from MLflow.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig
from src.repositories.functional_run_repository import search_runs


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
    # 1. Base MLflow fetch
    filter_string = "attributes.status = 'FINISHED' and tags.kind = 'model'"
    if base_filter:
        for k, v in base_filter.items():
            filter_string += f" and {k} = '{v}'"

    runs = search_runs(filter_string, output_format="list")

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


# ─────────────────────── groups signature ───────────────────────


def encode_groups_signature(groups_cfg) -> str:
    """
    Human-readable, deterministic signature of a groups discovery config.

    Format: ``group_by=rho,topology|k=3|filter=topology:torus``

    - ``group_by`` keys are sorted alphabetically.
    - ``k`` is sample_size as an integer, or ``"null"`` if unset.
    - ``filter`` is sorted ``key:value`` pairs; empty string if none.
    """
    group_by = sorted(groups_cfg.group_by)
    k = groups_cfg.sample_size if groups_cfg.sample_size is not None else "null"
    filter_dict = dict(groups_cfg.filter) if groups_cfg.filter else {}
    filter_str = ",".join(f"{fk}:{fv}" for fk, fv in sorted(filter_dict.items()))
    return f"group_by={','.join(group_by)}|k={k}|filter={filter_str}"


def decode_groups_signature(sig: str) -> dict:
    """
    Parse a groups signature string back to a plain dict.

    Returns ``{"group_by": [...], "sample_size": int | None, "filter": {...}}``.
    """
    parts = dict(p.split("=", 1) for p in sig.split("|"))
    raw_gb = parts.get("group_by", "")
    group_by = raw_gb.split(",") if raw_gb else []
    k_raw = parts.get("k", "null")
    sample_size = None if k_raw == "null" else int(k_raw)
    filter_raw = parts.get("filter", "")
    filter_dict: Dict[str, str] = {}
    if filter_raw:
        for pair in filter_raw.split(","):
            fk, fv = pair.split(":", 1)
            filter_dict[fk] = fv
    return {"group_by": group_by, "sample_size": sample_size, "filter": filter_dict}
