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


def _build_filter_and_postfilters(
    selector: Dict[str, Any],
) -> tuple[str, Dict[str, Any]]:
    """Split a declarative selector into an MLflow filter string and post-filters.

    MLflow only supports ``AND``-connected equality clauses — no ``OR``,
    parentheses, or ``IN``.  Predicates that require multi-value matching
    (``in``, ``range``) are returned as *post-filters* to be applied in
    Python after the query.

    Returns
    -------
    filter_string : str
        MLflow-compatible filter (equality clauses only).
    post_filters : dict
        ``{tag_key: set_of_allowed_string_values}`` for client-side filtering.
    """
    parts: list[str] = []
    post_filters: Dict[str, set] = {}
    for key, spec in selector.items():
        if isinstance(spec, dict):
            if "eq" in spec:
                parts.append(f"tags.{key} = '{spec['eq']}'")
            elif "range" in spec:
                lo, hi = spec["range"]
                vals = {str(v) for v in range(int(lo), int(hi) + 1)}
                post_filters[key] = vals
            elif "in" in spec:
                vals = {str(v) for v in spec["in"]}
                post_filters[key] = vals
        else:
            # Plain scalar → equality
            parts.append(f"tags.{key} = '{spec}'")
    return " and ".join(parts), post_filters


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
    selector_filter, post_filters = _build_filter_and_postfilters(selector)
    if selector_filter:
        full_filter = f"{base_filter} and {selector_filter}"
    else:
        full_filter = base_filter

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=full_filter,
        output_format="list",
    )

    # Apply client-side post-filters for in/range predicates
    if post_filters:
        filtered = []
        for r in runs:
            tags = r.data.tags
            if all(tags.get(k) in allowed for k, allowed in post_filters.items()):
                filtered.append(r)
        runs = filtered

    run_ids = sorted(r.info.run_id for r in runs)
    if not run_ids:
        raise ValueError(
            f"No FINISHED model runs match selector: {selector}\n"
            f"Filter: {full_filter}\n"
            f"Post-filters: {post_filters}"
        )
    return run_ids
