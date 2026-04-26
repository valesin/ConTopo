import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys, os

    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlflow"))
    from src.config.notebook import setup_environment
    from mlflow_helpers import get_ensemble_results_for_groups
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    GROUPS_OPTIONS = ["default", "samples3", "samples9", "samples9_mc"]
    return (
        GROUPS_OPTIONS,
        get_ensemble_results_for_groups,
        mo,
        plt,
        setup_environment,
    )


@app.cell
def _(setup_environment):
    cfg, experiment = setup_environment()
    return


@app.cell
def _(GROUPS_OPTIONS, mo):
    groups_a = mo.ui.dropdown(
        options=GROUPS_OPTIONS, value="samples9_mc", label="Groups A"
    )
    groups_b = mo.ui.dropdown(
        options=GROUPS_OPTIONS, value="samples3", label="Groups B"
    )
    vote_method = mo.ui.dropdown(
        options=["soft", "hard", "max_confidence", "conf_weighted"],
        value="soft",
        label="Vote method",
    )
    split = mo.ui.dropdown(options=["test", "val"], value="test", label="Split")
    mo.hstack([groups_a, groups_b, vote_method, split], gap=2)
    return groups_a, groups_b, split, vote_method


@app.cell
def _(
    get_ensemble_results_for_groups,
    groups_a,
    groups_b,
    mo,
    split,
    vote_method,
):
    def _load(groups_name):
        df = get_ensemble_results_for_groups(groups_name, split.value)
        if df.empty:
            return df
        return df[df["vote_method"] == vote_method.value].reset_index(drop=True)

    runs_a = _load(groups_a.value)
    runs_b = _load(groups_b.value)

    mo.stop(
        runs_a.empty and runs_b.empty,
        mo.callout(mo.md("No runs found for either groups config."), kind="warn"),
    )
    mo.md(
        f"**{groups_a.value}:** {len(runs_a)} runs &nbsp;|&nbsp; "
        f"**{groups_b.value}:** {len(runs_b)} runs"
    )
    return runs_a, runs_b


@app.cell
def _(groups_a, groups_b, plt, runs_a, runs_b, split, vote_method):
    import pandas as _pd

    def _agg(df):
        if df.empty:
            return _pd.DataFrame(columns=["rho_numeric", "mean_acc", "std_acc"])
        return (
            df.groupby("rho_numeric", sort=True)
            .agg(mean_acc=("accuracy", "mean"), std_acc=("accuracy", "std"))
            .fillna(0)
            .reset_index()
        )

    _agg_a = _agg(runs_a)
    _agg_b = _agg(runs_b)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for _label, _runs, _agg, _color, _ls in [
        (groups_a.value, runs_a, _agg_a, "#636EFA", "-"),
        (groups_b.value, runs_b, _agg_b, "#EF553B", "--"),
    ]:
        if _runs.empty:
            continue
        _rho = _agg["rho_numeric"].values
        _mean = _agg["mean_acc"].values
        _std = _agg["std_acc"].values
        ax.scatter(
            _runs["rho_numeric"],
            _runs["accuracy"],
            s=10,
            alpha=0.3,
            color=_color,
            zorder=2,
        )
        ax.fill_between(_rho, _mean - _std, _mean + _std, alpha=0.12, color=_color)
        ax.plot(
            _rho,
            _mean,
            marker="o",
            linestyle=_ls,
            color=_color,
            label=f"{_label} ({vote_method.value})",
            zorder=3,
        )

    ax.set_xlabel("ρ")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title(f"Ensemble accuracy — {split.value}")
    fig
    return


@app.cell
def _(groups_a, groups_b, mo, runs_a, runs_b):
    import pandas as _pd

    def _table(df, label):
        if df.empty:
            return _pd.DataFrame()
        return (
            df.groupby("rho_numeric")
            .agg(
                n=("accuracy", "count"),
                mean_acc=("accuracy", "mean"),
                std_acc=("accuracy", "std"),
            )
            .round(4)
            .reset_index()
            .assign(groups=label)[["groups", "rho_numeric", "n", "mean_acc", "std_acc"]]
        )

    mo.ui.table(
        _pd.concat([_table(runs_a, groups_a.value), _table(runs_b, groups_b.value)]),
        selection=None,
    )
    return


if __name__ == "__main__":
    app.run()
