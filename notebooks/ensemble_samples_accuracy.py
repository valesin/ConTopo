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
    from src.config.notebook import setup_environment, compose_groups
    from src.ensemble.selector import encode_groups_signature
    from src.repositories.functional_run_repository import search_runs
    import polars as pl
    import plotly.graph_objects as go

    return (
        compose_groups,
        encode_groups_signature,
        go,
        pl,
        search_runs,
        setup_environment,
    )


@app.cell
def _(mo):
    groups_override = mo.ui.dropdown(
        options=["samples9", "default"],
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
    return cfg, experiment


@app.cell
def _(compose_groups, encode_groups_signature, experiment, groups_override, mo):
    groups_cfg = compose_groups(groups_override.value)
    sig = encode_groups_signature(groups_cfg)
    mo.md(f"**Groups signature:** `{sig}`")
    return (sig,)


@app.cell
def _(mo, pl, search_runs, sig, split, vote_method):
    filter_str = (
        f"tags.kind = 'ensemble' and "
        f"tags.groups_signature = '{sig}' and "
        f"params.split = '{split.value}' and "
        f"params.method = '{vote_method.value}' and "
        f"attributes.status = 'FINISHED'"
    )
    runs_raw = search_runs(filter_str, output_format="pandas")

    if runs_raw.empty:
        mo.stop(
            True,
            mo.callout(mo.md("No runs found for this configuration."), kind="warn"),
        )

    runs = pl.from_pandas(runs_raw).with_columns(
        pl.col("tags.rho").cast(pl.Float64).alias("rho_numeric"),
        pl.col("metrics.ensemble_accuracy").alias("accuracy"),
        pl.col("metrics.comp_mean_acc").alias("comp_mean_acc"),
        (pl.col("metrics.ensemble_accuracy") - pl.col("metrics.comp_mean_acc")).alias(
            "gain"
        ),
    )
    mo.md(f"**Runs loaded:** {len(runs)}")
    return (runs,)


@app.cell
def _(pl, runs):
    agg = (
        runs.group_by("tags.rho", "rho_numeric")
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
def _(agg, go, vote_method):
    rho = agg["rho_numeric"].to_list()
    mean = agg["mean_acc"].to_list()
    std = agg["std_acc"].to_list()
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
def _(agg, mo, pl):
    table_df = agg.select(
        [
            pl.col("tags.rho").alias("ρ"),
            pl.col("n"),
            pl.col("mean_acc").round(4).alias("mean acc"),
            pl.col("std_acc").round(4).alias("std acc"),
            pl.col("mean_comp_acc").round(4).alias("comp mean acc"),
            pl.col("mean_gain").round(4).alias("gain (ens − comp)"),
        ]
    )
    mo.ui.table(table_df.to_pandas(), selection=None)
    return


if __name__ == "__main__":
    app.run()
