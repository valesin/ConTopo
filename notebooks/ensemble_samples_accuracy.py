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
    # Ensemble accuracy vs ρ — sampled combinations (samples9, soft, test)
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
def _(mo, setup_environment):
    cfg, experiment = setup_environment()
    mo.md(f"**Experiment:** `{experiment.name}`")
    return (experiment,)


@app.cell
def _(experiment, get_ensemble_results_for_groups, mo, pl):
    _GROUPS = "samples9"
    _SPLIT = "test"
    _METHOD = "soft"

    runs_pd = get_ensemble_results_for_groups(_GROUPS, experiment, _SPLIT)
    runs_pd = runs_pd[runs_pd["vote_method"] == _METHOD]

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
            pl.col("gain").std().alias("std_gain"),
            pl.len().alias("n"),
        )
        .sort("rho_numeric")
    )
    return (agg,)


@app.cell
def _(mo):
    mo.md(
        """
    ### Ensemble vs solo component accuracy
    """
    )
    return


@app.cell
def _(agg, go, pl):
    rho = agg["rho"].to_list()
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
                name="Ensemble acc (soft, mean over combos)",
            ),
            go.Scatter(
                x=rho,
                y=agg["mean_comp_acc"].to_list(),
                mode="lines+markers",
                line=dict(color="rgb(239,85,59)", dash="dot"),
                name="Solo component acc (mean over all N models in group)",
            ),
        ]
    )
    fig.update_layout(
        xaxis_title="ρ",
        yaxis_title="Accuracy",
        xaxis=dict(type="category"),
        template="simple_white",
        legend=dict(orientation="h", y=1.1),
    )
    fig
    return


@app.cell
def _(mo):
    mo.md(
        """
    ### Does gain increase with ρ?
    """
    )
    return


@app.cell
def _(agg, go, pl):
    _rho = agg["rho"].to_list()
    _gain = agg["mean_gain"].to_list()
    _std = agg.with_columns(pl.col("std_gain").fill_null(0.0))["std_gain"].to_list()
    _upper = [g + s for g, s in zip(_gain, _std)]
    _lower = [g - s for g, s in zip(_gain, _std)]

    _fig = go.Figure(
        [
            go.Scatter(
                x=_rho + _rho[::-1],
                y=_upper + _lower[::-1],
                fill="toself",
                fillcolor="rgba(50,171,96,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=False,
            ),
            go.Scatter(
                x=_rho,
                y=_gain,
                mode="lines+markers",
                line=dict(color="rgb(50,171,96)"),
                name="Mean gain (ensemble − solo component)",
            ),
        ]
    )
    _fig.update_layout(
        xaxis_title="ρ",
        yaxis_title="Gain",
        xaxis=dict(type="category"),
        template="simple_white",
        legend=dict(orientation="h", y=1.1),
    )
    _fig
    return


@app.cell
def _(mo, runs):
    from scipy.stats import spearmanr

    _rho_vals = runs["rho_numeric"].to_list()
    _gain_vals = runs["gain"].to_list()
    _stat, _pval = spearmanr(_rho_vals, _gain_vals)

    mo.callout(
        mo.md(
            f"**Spearman ρ (rho_numeric vs gain) = {_stat:.3f}** &nbsp;|&nbsp; "
            f"p = {_pval:.2e} &nbsp;|&nbsp; n = {len(_gain_vals)} combos"
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ### Confounding check — is gain driven by weak components?

    If higher gain merely reflects weaker solo components (easier to beat), the scatter
    below would show a negative slope regardless of ρ. Colour by ρ to see whether the
    gain-vs-solo relationship shifts across regularisation levels.
    """
    )
    return


@app.cell
def _(go, runs):
    import plotly.colors as pc

    _rho_vals = runs["rho"].unique().sort().to_list()
    _palette = pc.qualitative.Plotly

    _fig = go.Figure()
    for _i, _r in enumerate(_rho_vals):
        _subset = runs.filter(runs["rho"] == _r)
        _fig.add_trace(
            go.Scatter(
                x=_subset["comp_mean_acc"].to_list(),
                y=_subset["gain"].to_list(),
                mode="markers",
                marker=dict(color=_palette[_i % len(_palette)], opacity=0.5, size=6),
                name=f"ρ = {_r}",
            )
        )

    _fig.update_layout(
        xaxis_title="Solo component acc (this combo)",
        yaxis_title="Gain (ensemble − solo component)",
        template="simple_white",
        legend=dict(title="ρ", orientation="v"),
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        """
    ---
    ### Reference tables
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    **Aggregated by ρ** — solo component acc is derived by averaging `comp_mean_acc` across
    all C(N, k) combos; by combinatorial symmetry this equals the mean acc of all N models
    in the pool.
    """
    )
    return


@app.cell
def _(agg, mo, pl):
    table_df = agg.select(
        [
            pl.col("rho").alias("ρ"),
            pl.col("n").alias("# combos"),
            pl.col("mean_acc").round(4).alias("mean ensemble acc"),
            pl.col("std_acc").round(4).alias("std ensemble acc"),
            pl.col("mean_comp_acc")
            .round(4)
            .alias("solo component acc (mean of all N models)"),
            pl.col("mean_gain").round(4).alias("mean gain (ensemble − solo component)"),
        ]
    )
    mo.ui.table(table_df.to_pandas(), selection=None)
    return


@app.cell
def _(mo):
    mo.md(
        """
    **Per combo** — `solo component acc (this combo)` is `comp_mean_acc` as logged by the
    pipeline: mean accuracy of the k solo models in this specific combo.
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
                pl.col("ensemble_name").alias("combo"),
                pl.col("accuracy").round(4).alias("ensemble acc"),
                pl.col("comp_mean_acc")
                .round(4)
                .alias("solo component acc (this combo)"),
                pl.col("gain").round(4).alias("gain vs own solos"),
                (pl.col("accuracy") - pl.col("group_comp_acc"))
                .round(4)
                .alias("gain vs group solo mean"),
            ]
        )
    )
    mo.ui.table(per_run.to_pandas(), selection=None)
    return (per_run,)


@app.cell
def _(mo, per_run):
    _df = mo.sql(
        f"""
        SELECT rho, avg("gain vs own solos"), avg("gain vs group solo mean") FROM per_run
        GROUP BY rho
        ORDER BY rho
        """
    )
    return


if __name__ == "__main__":
    app.run()
