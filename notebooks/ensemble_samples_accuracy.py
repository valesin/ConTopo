import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
    # Ensemble accuracy vs ρ — sampled combinations
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
    import plotly.graph_objects as go

    return get_ensemble_results_for_groups, go, pl, setup_environment


@app.cell
def _(mo):
    groups_override = mo.ui.dropdown(
        options=["samples9", "samples3", "default"],
        value="samples9",
        label="Groups config",
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
    mo.hstack([groups_override, vote_method, split], gap=2)
    return groups_override, split, vote_method


@app.cell
def _(mo, setup_environment):
    cfg, experiment = setup_environment()
    mo.md(f"**Experiment:** `{experiment.name}`")
    return (experiment,)


@app.cell
def _(
    experiment,
    get_ensemble_results_for_groups,
    groups_override,
    mo,
    pl,
    split,
    vote_method,
):
    runs_pd = get_ensemble_results_for_groups(
        groups_override.value, experiment, split.value
    )
    runs_pd = runs_pd[runs_pd["vote_method"] == vote_method.value]

    if runs_pd.empty:
        mo.stop(
            True,
            mo.callout(mo.md("No runs found for this configuration."), kind="warn"),
        )

    runs = pl.from_pandas(runs_pd).with_columns(
        (pl.col("accuracy") - pl.col("comp_mean_acc")).alias("gain"),
    )
    mo.md(f"**Runs loaded:** {len(runs)}")
    return (runs,)


@app.cell
def _(pl, runs):
    agg = (
        runs.group_by("rho", "rho_numeric")
        .agg(
            pl.col("accuracy").mean().alias("mean_acc"),
            pl.col("accuracy").std().alias("std_acc"),
            pl.col("comp_mean_acc").mean().alias("mean_comp_acc"),
            pl.col("gain").mean().alias("mean_gain"),
            pl.len().alias("n"),
        )
        .sort("rho_numeric")
    )
    return (agg,)


@app.cell
def _(agg, go, pl, vote_method):
    rho = agg["rho_numeric"].to_list()
    mean = agg["mean_acc"].to_list()
    std = agg.with_columns(pl.col("std_acc").fill_null(0.0))["std_acc"].to_list()
    upper = [m + s for m, s in zip(mean, std)]
    lower = [m - s for m, s in zip(mean, std)]

    fig = go.Figure(
        [
            go.Scatter(
                x=rho + rho[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor="rgba(99,110,250,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=False,
            ),
            go.Scatter(
                x=rho,
                y=mean,
                mode="lines+markers",
                line=dict(color="rgb(99,110,250)"),
                name=f"Ensemble ({vote_method.value})",
            ),
            go.Scatter(
                x=rho,
                y=agg["mean_comp_acc"].to_list(),
                mode="lines+markers",
                line=dict(color="rgb(239,85,59)", dash="dot"),
                name="Component mean",
            ),
        ]
    )
    fig.update_layout(
        xaxis_title="ρ",
        yaxis_title="Accuracy",
        template="simple_white",
        legend=dict(orientation="h", y=1.1),
    )
    fig
    return


@app.cell
def _(mo):
    mo.md(
        """
    ### Aggregated by ρ — gain vs mean component accuracy across all ensembles in the group
    """
    )
    return


@app.cell
def _(agg, mo, pl):
    table_df = agg.select(
        [
            pl.col("rho").alias("ρ"),
            pl.col("n").alias("# ensembles"),
            pl.col("mean_acc").round(4).alias("mean ensemble acc"),
            pl.col("std_acc").round(4).alias("std ensemble acc"),
            pl.col("mean_comp_acc").round(4).alias("mean of component means"),
            pl.col("mean_gain").round(4).alias("mean gain (ens − comp mean)"),
        ]
    )
    mo.ui.table(table_df.to_pandas(), selection=None)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ### Per ensemble — gain vs own components and vs the fixed group mean
    """
    )
    return


@app.cell
def _(agg, mo, pl, runs):
    per_run = (
        runs.join(
            agg.select(
                ["rho_numeric", pl.col("mean_comp_acc").alias("group_comp_acc")]
            ),
            on="rho_numeric",
            how="left",
        )
        .sort("rho_numeric")
        .select(
            [
                pl.col("rho"),
                pl.col("ensemble_name").alias("ensemble"),
                pl.col("accuracy").round(4).alias("ensemble acc"),
                pl.col("comp_mean_acc").round(4).alias("mean acc of own components"),
                pl.col("gain").round(4).alias("gain vs own components"),
                (pl.col("accuracy") - pl.col("group_comp_acc"))
                .round(4)
                .alias("gain vs group component mean"),
            ]
        )
    )
    mo.ui.table(per_run.to_pandas(), selection=None)
    return (per_run,)


@app.cell
def _(mo, per_run):
    _df = mo.sql(
        f"""
        SELECT rho, avg("gain vs own components"), avg("gain vs group component mean") FROM per_run
        GROUP BY rho
        """
    )
    return


if __name__ == "__main__":
    app.run()
