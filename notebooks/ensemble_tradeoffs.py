import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import sys, os

    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlflow"))
    from src.config.notebook import setup_environment
    from mlflow_helpers import (
        get_runs,
        get_ensemble_results_for_groups,
        get_consistency_results_for_groups,
        get_expected_component_hashes,
    )
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    DIVERSITY_METRICS = [
        "q_statistic",
        "disagreement",
        "double_fault",
        "correlation",
        "interrater_agreement",
        "iou_top_n",
    ]
    return (
        DIVERSITY_METRICS,
        cm,
        get_consistency_results_for_groups,
        get_ensemble_results_for_groups,
        get_expected_component_hashes,
        get_runs,
        mo,
        pd,
        plt,
        setup_environment,
    )


@app.cell
def _(setup_environment):
    cfg, experiment = setup_environment()
    return


@app.cell
def _(DIVERSITY_METRICS, mo):
    groups_ui = mo.ui.dropdown(
        options=["default", "samples3", "samples9", "samples9_mc"],
        value="samples9_mc",
        label="Groups config",
    )
    split_ui = mo.ui.dropdown(options=["test", "val"], value="test", label="Split")
    vote_method_ui = mo.ui.dropdown(
        options=["soft", "hard", "max_confidence", "conf_weighted"],
        value="soft",
        label="Vote method",
    )
    diversity_metric_ui = mo.ui.dropdown(
        options=DIVERSITY_METRICS,
        value="disagreement",
        label="Diversity metric",
    )
    mo.hstack([groups_ui, split_ui, vote_method_ui, diversity_metric_ui], gap=2)
    return diversity_metric_ui, groups_ui, split_ui, vote_method_ui


@app.cell
def _(
    diversity_metric_ui,
    get_consistency_results_for_groups,
    get_ensemble_results_for_groups,
    get_expected_component_hashes,
    get_runs,
    groups_ui,
    mo,
    pd,
    split_ui,
    vote_method_ui,
):
    _ens_all = get_ensemble_results_for_groups(groups_ui.value, split_ui.value)
    ens = _ens_all[_ens_all["vote_method"] == vote_method_ui.value].reset_index(
        drop=True
    )

    con = get_consistency_results_for_groups(groups_ui.value, split_ui.value)

    _hashes = get_expected_component_hashes(groups_ui.value)
    _div_all = get_runs(
        "diversity",
        split=split_ui.value,
        diversity_metric=diversity_metric_ui.value,
    )
    _div_filtered = _div_all[_div_all["tags.component_set_hash"].isin(_hashes)]
    div = pd.DataFrame(
        {
            "cs_hash": _div_filtered["tags.component_set_hash"].values,
            "diversity_value": pd.to_numeric(
                _div_filtered[f"metrics.{diversity_metric_ui.value}"], errors="coerce"
            ).values,
        }
    )

    mo.stop(
        ens.empty or con.empty or div.empty,
        mo.callout(
            mo.md(
                "No data found for one or more run kinds. Check groups config and split."
            ),
            kind="warn",
        ),
    )
    mo.md(
        f"ensemble: **{len(ens)}** | consistency: **{len(con)}** | diversity: **{len(div)}**"
    )
    return con, div, ens


@app.cell
def _(con, div, ens, mo):
    joined = (
        ens[["cs_hash", "rho", "rho_numeric", "accuracy"]]
        .merge(con[["cs_hash", "mean_rsa_correlation"]], on="cs_hash", how="inner")
        .merge(div, on="cs_hash", how="inner")
        .dropna()
        .reset_index(drop=True)
    )

    mo.stop(
        len(joined) == 0,
        mo.callout(
            mo.md("Join produced no rows — cs_hash mismatch across run kinds."),
            kind="warn",
        ),
    )
    mo.md(
        f"Joined **{len(joined)} combinations** across "
        f"**{joined['rho'].nunique()} ρ values**."
    )
    return (joined,)


@app.cell
def _(cm, diversity_metric_ui, groups_ui, joined, plt, split_ui):
    _rho = joined["rho_numeric"].values
    _acc = joined["accuracy"].values
    _con = joined["mean_rsa_correlation"].values
    _div = joined["diversity_value"].values

    _norm = plt.Normalize(_rho.min(), _rho.max())
    _cmap = cm.viridis
    _colors = _cmap(_norm(_rho))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    _panels = [
        (_div, _acc, diversity_metric_ui.value, "accuracy"),
        (_con, _acc, "mean RSA correlation", "accuracy"),
        (_div, _con, diversity_metric_ui.value, "mean RSA correlation"),
    ]
    for ax, (x, y, xlabel, ylabel) in zip(axes, _panels):
        ax.scatter(x, y, c=_colors, s=25, alpha=0.7, zorder=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    _sm = cm.ScalarMappable(cmap=_cmap, norm=_norm)
    fig.colorbar(_sm, ax=axes, label="ρ", fraction=0.02, pad=0.02, shrink=0.8)
    fig.suptitle(
        f"Ensemble tradeoffs — {diversity_metric_ui.value} / {groups_ui.value} / {split_ui.value}",
        fontsize=13,
    )
    fig
    return


if __name__ == "__main__":
    app.run()
