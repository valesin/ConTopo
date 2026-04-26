import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys, os

    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlflow"))
    from src.config.notebook import setup_environment
    from mlflow_helpers import (
        get_runs,
        get_consistency_results_for_groups,
        get_expected_component_hashes,
    )
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    METRIC_MAP = {
        "mean_rsa_correlation": ("consistency", "mean_rsa_correlation"),
        "q_statistic": ("diversity", "metrics.q_statistic"),
        "disagreement": ("diversity", "metrics.disagreement"),
        "double_fault": ("diversity", "metrics.double_fault"),
        "correlation": ("diversity", "metrics.correlation"),
        "interrater_agreement": ("diversity", "metrics.interrater_agreement"),
        "iou_top_n": ("diversity", "metrics.iou_top_n"),
    }
    return (
        METRIC_MAP,
        get_consistency_results_for_groups,
        get_expected_component_hashes,
        get_runs,
        mo,
        np,
        pd,
        plt,
        setup_environment,
    )


@app.cell
def _(setup_environment):
    cfg, experiment = setup_environment()
    return


@app.cell
def _(METRIC_MAP, mo):
    groups_ui = mo.ui.dropdown(
        options=["default", "samples3", "samples9", "samples9_mc"],
        value="samples9_mc",
        label="Groups config",
    )
    split_ui = mo.ui.dropdown(options=["test", "val"], value="test", label="Split")
    diagnostic = mo.ui.dropdown(
        options=list(METRIC_MAP.keys()),
        value="mean_rsa_correlation",
        label="Diagnostic",
    )
    mo.hstack([groups_ui, split_ui, diagnostic], gap=2)
    return diagnostic, groups_ui, split_ui


@app.cell
def _(
    METRIC_MAP,
    diagnostic,
    get_consistency_results_for_groups,
    get_expected_component_hashes,
    get_runs,
    groups_ui,
    mo,
    pd,
    split_ui,
):
    _kind, _col = METRIC_MAP[diagnostic.value]

    if _kind == "consistency":
        _raw = get_consistency_results_for_groups(groups_ui.value, split_ui.value)
        ens_runs = (
            _raw[["rho", "rho_numeric", _col]]
            .rename(columns={_col: "metric"})
            .reset_index(drop=True)
        )
    else:
        _hashes = get_expected_component_hashes(groups_ui.value)
        _all = get_runs("diversity", split=split_ui.value)
        _filtered = _all[_all["tags.component_set_hash"].isin(_hashes)]
        ens_runs = pd.DataFrame(
            {
                "rho": _filtered["params.rho"].values,
                "metric": pd.to_numeric(_filtered[_col], errors="coerce").values,
            }
        )
        ens_runs["rho_numeric"] = pd.to_numeric(ens_runs["rho"], errors="coerce")
        ens_runs = ens_runs.sort_values("rho_numeric").reset_index(drop=True)

    mo.stop(
        len(ens_runs) == 0,
        mo.callout(mo.md("No runs found for this configuration."), kind="warn"),
    )
    mo.md(
        f"Loaded **{len(ens_runs)} runs** across **{ens_runs['rho'].nunique()} ρ values**."
    )
    return (ens_runs,)


@app.cell
def _(ens_runs, mo):
    mo.sql(
        """
        SELECT rho, count(*) AS n
        FROM ens_runs
        GROUP BY rho
        ORDER BY CAST(rho AS DOUBLE)
        """
    )
    return


@app.cell
def _(diagnostic, ens_runs, groups_ui, np, plt, split_ui):
    _rho = ens_runs["rho_numeric"].values
    _metric = ens_runs["metric"].values
    _unique_rhos = np.sort(np.unique(_rho))
    _means = np.array([_metric[_rho == r].mean() for r in _unique_rhos])
    _stds = np.array([_metric[_rho == r].std() for r in _unique_rhos])

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.fill_between(
        _unique_rhos, _means - _stds, _means + _stds, alpha=0.15, color="steelblue"
    )
    ax.scatter(_rho, _metric, s=15, alpha=0.5, color="steelblue", zorder=2)
    ax.plot(_unique_rhos, _means, color="firebrick", zorder=3)
    ax.set_xlabel("ρ")
    ax.set_ylabel(diagnostic.value)
    ax.set_title(f"ρ vs {diagnostic.value} — {groups_ui.value} / {split_ui.value}")
    fig
    return


if __name__ == "__main__":
    app.run()
