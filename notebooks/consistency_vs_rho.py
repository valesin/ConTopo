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
    # Consistency vs ρ — sampled combinations (samples9_mc, test)
    """
    )
    return


@app.cell
def _():
    import sys, os

    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlflow"))
    from src.config.notebook import setup_environment
    from mlflow_helpers import get_consistency_results_for_groups, varying_fields
    import polars as pl
    import altair as alt

    return (
        alt,
        get_consistency_results_for_groups,
        pl,
        setup_environment,
        varying_fields,
    )


@app.cell
def _(mo, setup_environment):
    cfg, experiment = setup_environment()
    mo.md(f"**Experiment:** `{experiment.name}`")
    return


@app.cell
def _(get_consistency_results_for_groups, mo, pl):
    _GROUPS = "samples9_mc"
    _SPLIT = "test"

    runs_pd = get_consistency_results_for_groups(_GROUPS, _SPLIT)

    if runs_pd.empty:
        mo.stop(
            True,
            mo.callout(mo.md("No runs found for this configuration."), kind="warn"),
        )

    runs = pl.from_pandas(runs_pd)
    mo.md(f"**Runs loaded:** {len(runs)}")
    return (runs,)


@app.cell
def _(pl, runs):
    agg = (
        runs.group_by("rho", "rho_numeric")
        .agg(
            pl.col("mean_rsa_correlation").mean().alias("mean_cons"),
            pl.col("mean_rsa_correlation").std().alias("std_cons"),
            pl.len().alias("n"),
        )
        .sort("rho_numeric")
    )
    return (agg,)


@app.cell
def _(mo):
    mo.md(
        """
    ### Mean RSA correlation per ρ
    """
    )
    return


@app.cell
def _(agg, alt, pl, runs):
    _agg_df = (
        agg.with_columns(pl.col("std_cons").fill_null(0.0))
        .select(
            pl.col("rho"),
            pl.col("rho_numeric"),
            pl.col("mean_cons"),
            (pl.col("mean_cons") - pl.col("std_cons")).alias("lower"),
            (pl.col("mean_cons") + pl.col("std_cons")).alias("upper"),
        )
        .to_pandas()
    )

    _pts_df = runs.select(
        pl.col("rho"), pl.col("rho_numeric"), pl.col("mean_rsa_correlation")
    ).to_pandas()

    _sort = alt.EncodingSortField(field="rho_numeric")
    _yscale = alt.Scale(zero=False)

    _band = (
        alt.Chart(_agg_df)
        .mark_area(opacity=0.15, color="#636EFA")
        .encode(
            x=alt.X("rho:O", sort=_sort, title="ρ"),
            y=alt.Y("lower:Q", title="Mean RSA correlation", scale=_yscale),
            y2=alt.Y2("upper:Q"),
        )
    )

    _line = (
        alt.Chart(_agg_df)
        .mark_line(point=True, color="#636EFA")
        .encode(
            x=alt.X("rho:O", sort=_sort),
            y=alt.Y("mean_cons:Q", title="Mean RSA correlation", scale=_yscale),
            tooltip=["rho", "mean_cons"],
        )
    )

    _points = (
        alt.Chart(_pts_df)
        .mark_circle(size=30, color="#1a1a2e", opacity=0.5)
        .encode(
            x=alt.X("rho:O", sort=_sort),
            y=alt.Y("mean_rsa_correlation:Q", scale=_yscale),
            tooltip=["rho", "mean_rsa_correlation"],
        )
    )

    (_band + _line + _points).resolve_scale(y="shared").properties(
        width=600, height=350, title="Model consistency (mean RSA correlation) vs ρ"
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ### Reference table
    """
    )
    return


@app.cell
def _(agg, mo, pl):
    mo.ui.table(
        agg.select(
            pl.col("rho").alias("ρ"),
            pl.col("n").alias("# combos"),
            pl.col("mean_cons").round(4).alias("mean RSA correlation"),
            pl.col("std_cons").round(4).alias("std RSA correlation"),
        ).to_pandas(),
        selection=None,
    )
    return


if __name__ == "__main__":
    app.run()
