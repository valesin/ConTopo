"""
Declarative ensemble component selector (Tier-1).

Resolves an ensemble definition from ``conf/ensemble/*.yaml`` to a deterministic
sorted list of MLflow run IDs by querying MLflow with declarative filters.

Supported selector predicates:
  - ``eq``    : exact equality  (e.g. ``rho: {eq: 0}``)
  - ``range`` : inclusive range (e.g. ``trial: {range: [0, 9]}``)
  - ``in``    : membership list (e.g. ``trial: {in: [0, 2, 5]}``)
"""

from __future__ import annotations

from typing import Any, Dict, List

import mlflow


def _build_filter(selector: Dict[str, Any]) -> str:
    """Convert a declarative selector dict into an MLflow filter string."""
    parts: list[str] = []
    for key, spec in selector.items():
        if isinstance(spec, dict):
            if "eq" in spec:
                parts.append(f"tags.{key} = '{spec['eq']}'")
            elif "range" in spec:
                lo, hi = spec["range"]
                # Range over integer tags — expand to OR.  MLflow doesn't
                # support numeric range queries on tags, so we enumerate.
                vals = list(range(int(lo), int(hi) + 1))
                or_parts = " or ".join(f"tags.{key} = '{v}'" for v in vals)
                parts.append(f"({or_parts})")
            elif "in" in spec:
                or_parts = " or ".join(f"tags.{key} = '{v}'" for v in spec["in"])
                parts.append(f"({or_parts})")
        else:
            # Plain scalar → equality
            parts.append(f"tags.{key} = '{spec}'")
    return " and ".join(parts)


def resolve_components(
    selector: Dict[str, Any],
    experiment_name: str,
) -> List[str]:
    """
    Query MLflow for FINISHED model runs matching ``selector``.

    Returns a **sorted** list of run IDs.
    """
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"MLflow experiment '{experiment_name}' not found.")

    base_filter = "attributes.status = 'FINISHED' and tags.kind = 'model'"
    selector_filter = _build_filter(selector)
    if selector_filter:
        full_filter = f"{base_filter} and {selector_filter}"
    else:
        full_filter = base_filter

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=full_filter,
        output_format="list",
    )
    run_ids = sorted(r.info.run_id for r in runs)
    if not run_ids:
        raise ValueError(
            f"No FINISHED model runs match selector: {selector}\n"
            f"Filter: {full_filter}"
        )
    return run_ids
