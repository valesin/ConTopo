import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
    # Ensemble accuracy comparison — groups A vs B
    """
    )
    return


@app.cell
def _():
    import sys, os

    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlflow"))
    from src.config.notebook import setup_environment
    from mlflow_helpers import get_ensemble_results_for_groups
    import polars as pl

    # import plotly.graphmakes t_objects as go
    return get_ensemble_results_for_groups, pl, setup_environment


@app.cell
def _(mo):
    groups_a = mo.ui.dropdown(
        options=["samples9", "samples3", "default"],
        value="samples9",
        label="Groups A",
    )
    groups_b = mo.ui.dropdown(
        options=["samples9", "samples3", "default"],
        value="samples3",
        label="Groups B",
    )
    vote_method = mo.ui.dropdown(
        options=["soft", "hard", "max_confidence", "conf_weighted"],
        value="soft",
        label="Vote method",
    )
    split = mo.ui.dropdown(
        options=["test", "val"],
        value="test",
        label="Split",
    )
    mo.hstack([groups_a, groups_b, vote_method, split], gap=2)
    return groups_a, groups_b, split, vote_method


@app.cell
def _(mo, setup_environment):
    cfg, experiment = setup_environment()
    mo.md(f"**Experiment:** `{experiment.name}`")
    return


@app.cell
def _(
    get_ensemble_results_for_groups,
    groups_a,
    groups_b,
    mo,
    pl,
    split,
    vote_method,
):
    def _load(groups_name):
        runs_pd = get_ensemble_results_for_groups(groups_name, split.value)
        runs_pd = runs_pd[runs_pd["vote_method"] == vote_method.value]
        return pl.from_pandas(runs_pd).with_columns(
            (pl.col("accuracy") - pl.col("comp_mean_acc")).alias("gain"),
        )

    runs_a = _load(groups_a.value)
    runs_b = _load(groups_b.value)

    if runs_a.is_empty() and runs_b.is_empty():
        mo.stop(
            True,
            mo.callout(mo.md("No runs found for either groups config."), kind="warn"),
        )

    mo.md(
        f"**{groups_a.value}:** {len(runs_a)} runs &nbsp;|&nbsp; "
        f"**{groups_b.value}:** {len(runs_b)} runs"
    )
    return runs_a, runs_b


@app.cell
def _(pl, runs_a, runs_b):
    def _agg(runs):
        if runs.is_empty():
            return pl.DataFrame(
                schema={
                    "rho": pl.Utf8,
                    "rho_numeric": pl.Float64,
                    "mean_acc": pl.Float64,
                    "std_acc": pl.Float64,
                    "mean_comp_acc": pl.Float64,
                    "n": pl.UInt32,
                }
            )
        return (
            runs.group_by("rho", "rho_numeric")
            .agg(
                pl.col("accuracy").mean().alias("mean_acc"),
                pl.col("accuracy").std().alias("std_acc"),
                pl.col("comp_mean_acc").mean().alias("mean_comp_acc"),
                pl.len().alias("n"),
            )
            .sort("rho_numeric")
        )

    agg_a = _agg(runs_a)
    agg_b = _agg(runs_b)
    return agg_a, agg_b


@app.cell
def _(agg_a, agg_b, go, groups_a, groups_b, vote_method):
    COLOR_A = "rgb(99,110,250)"
    COLOR_B = "rgb(239,85,59)"

    def _band(agg, color):
        rho = agg["rho_numeric"].to_list()
        mean = agg["mean_acc"].to_list()
        std = [s or 0.0 for s in agg["std_acc"].to_list()]
        upper = [m + s for m, s in zip(mean, std)]
        lower = [m - s for m, s in zip(mean, std)]
        r, g, b = color[4:-1].split(",")
        fill = f"rgba({r},{g},{b},0.15)"
        return rho, mean, upper, lower, fill

    traces = []
    if not agg_a.is_empty():
        rho_a, mean_a, upper_a, lower_a, fill_a = _band(agg_a, COLOR_A)
        traces += [
            go.Scatter(
                x=rho_a + rho_a[::-1],
                y=upper_a + lower_a[::-1],
                fill="toself",
                fillcolor=fill_a,
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=False,
            ),
            go.Scatter(
                x=rho_a,
                y=mean_a,
                mode="lines+markers",
                line=dict(color=COLOR_A),
                name=f"{groups_a.value} ({vote_method.value})",
            ),
        ]

    if not agg_b.is_empty():
        rho_b, mean_b, upper_b, lower_b, fill_b = _band(agg_b, COLOR_B)
        traces += [
            go.Scatter(
                x=rho_b + rho_b[::-1],
                y=upper_b + lower_b[::-1],
                fill="toself",
                fillcolor=fill_b,
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=False,
            ),
            go.Scatter(
                x=rho_b,
                y=mean_b,
                mode="lines+markers",
                line=dict(color=COLOR_B, dash="dash"),
                name=f"{groups_b.value} ({vote_method.value})",
            ),
        ]

    fig = go.Figure(traces)
    fig.update_layout(
        xaxis_title="ρ",
        yaxis_title="Accuracy",
        template="simple_white",
        legend=dict(orientation="h", y=1.1),
    )
    fig
    return


@app.cell
def _(agg_a, agg_b, groups_a, groups_b, mo, pl):
    def _table(agg, label):
        if agg.is_empty():
            return pl.DataFrame(
                {"groups": [], "ρ": [], "n": [], "mean acc": [], "std acc": []}
            )
        return agg.select(
            [
                pl.lit(label).alias("groups"),
                pl.col("rho").alias("ρ"),
                pl.col("n"),
                pl.col("mean_acc").round(4).alias("mean acc"),
                pl.col("std_acc").round(4).alias("std acc"),
            ]
        )

    combined = pl.concat([_table(agg_a, groups_a.value), _table(agg_b, groups_b.value)])
    mo.ui.table(combined.to_pandas(), selection=None)
    return


if __name__ == "__main__":
    app.run()
