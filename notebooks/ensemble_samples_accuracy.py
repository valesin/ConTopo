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
    import altair as alt

    return alt, get_ensemble_results_for_groups, pl, setup_environment


@app.cell
def _(mo, setup_environment):
    cfg, experiment = setup_environment()
    mo.md(f"**Experiment:** `{experiment.name}`")
    return


@app.cell
def _(get_ensemble_results_for_groups, mo, pl):
    _GROUPS = "samples9_mc"
    _SPLIT = "test"
    _METHOD = "soft"

    runs_pd = get_ensemble_results_for_groups(_GROUPS, _SPLIT)
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
def _(agg, alt, pl):
    _df = (
        agg.with_columns(pl.col("std_acc").fill_null(0.0))
        .select(
            pl.col("rho"),
            pl.col("rho_numeric"),
            pl.col("mean_acc"),
            (pl.col("mean_acc") - pl.col("std_acc")).alias("lower"),
            (pl.col("mean_acc") + pl.col("std_acc")).alias("upper"),
            pl.col("mean_comp_acc"),
        )
        .to_pandas()
    )

    _sort = alt.EncodingSortField(field="rho_numeric")

    _yscale = alt.Scale(zero=False)

    _band = (
        alt.Chart(_df)
        .mark_area(opacity=0.15, color="#636EFA")
        .encode(
            x=alt.X("rho:O", sort=_sort, title="ρ"),
            y=alt.Y("lower:Q", title="Accuracy", scale=_yscale),
            y2=alt.Y2("upper:Q"),
        )
    )
    _ensemble = (
        alt.Chart(_df)
        .mark_line(point=True, color="#636EFA")
        .encode(
            x=alt.X("rho:O", sort=_sort),
            y=alt.Y("mean_acc:Q", title="Accuracy", scale=_yscale),
            tooltip=["rho", "mean_acc"],
        )
        .properties(title="Ensemble acc (soft, mean over combos)")
    )
    _solo = (
        alt.Chart(_df)
        .mark_line(point=True, color="#EF553B", strokeDash=[4, 2])
        .encode(
            x=alt.X("rho:O", sort=_sort),
            y=alt.Y("mean_comp_acc:Q", title="Accuracy", scale=_yscale),
            tooltip=["rho", "mean_comp_acc"],
        )
        .properties(title="Mean component acc (avg. over all N models in group)")
    )

    (_band + _ensemble + _solo).resolve_scale(y="shared").properties(width=600)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ### Mean accuracy per ρ (soft voting)
    """
    )
    return


@app.cell
def _(agg, alt, pl, runs):
    _bar_df = agg.select(
        pl.col("rho"), pl.col("rho_numeric"), pl.col("mean_acc")
    ).to_pandas()

    _pts_df = runs.select(
        pl.col("rho"), pl.col("rho_numeric"), pl.col("accuracy")
    ).to_pandas()

    _floor = round(float(_pts_df["accuracy"].min()) - 0.005, 3)
    _sort = alt.EncodingSortField(field="rho_numeric")
    _yscale = alt.Scale(zero=False, domainMin=_floor)

    _bars = (
        alt.Chart(_bar_df)
        .mark_bar(opacity=0.55, color="#636EFA")
        .encode(
            x=alt.X("rho:O", sort=_sort, title="ρ"),
            y=alt.Y("mean_acc:Q", title="Accuracy", scale=_yscale),
            y2=alt.Y2(datum=_floor),
            tooltip=["rho", alt.Tooltip("mean_acc:Q", format=".4f")],
        )
    )

    _points = (
        alt.Chart(_pts_df)
        .mark_circle(size=55, color="#1a1a2e", opacity=0.75)
        .encode(
            x=alt.X("rho:O", sort=_sort),
            y=alt.Y("accuracy:Q", scale=_yscale),
            tooltip=["rho", alt.Tooltip("accuracy:Q", format=".4f")],
        )
    )

    (_bars + _points).properties(
        width=400,
        height=320,
        title="Mean ensemble accuracy per ρ — soft voting (bars = mean, dots = individual combos)",
    )
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
def _(agg, alt, pl):
    _df = (
        agg.with_columns(pl.col("std_gain").fill_null(0.0))
        .select(
            pl.col("rho"),
            pl.col("rho_numeric"),
            pl.col("mean_gain"),
            (pl.col("mean_gain") - pl.col("std_gain")).alias("lower"),
            (pl.col("mean_gain") + pl.col("std_gain")).alias("upper"),
        )
        .to_pandas()
    )

    _sort = alt.EncodingSortField(field="rho_numeric")

    _yscale = alt.Scale(zero=False)

    _band = (
        alt.Chart(_df)
        .mark_area(opacity=0.15, color="#32AB60")
        .encode(
            x=alt.X("rho:O", sort=_sort, title="ρ"),
            y=alt.Y("lower:Q", title="Gain", scale=_yscale),
            y2=alt.Y2("upper:Q"),
        )
    )
    _line = (
        alt.Chart(_df)
        .mark_line(point=True, color="#32AB60")
        .encode(
            x=alt.X("rho:O", sort=_sort, title="ρ"),
            y=alt.Y("mean_gain:Q", title="Gain", scale=_yscale),
            tooltip=["rho", "mean_gain"],
        )
        .properties(title="Mean gain (ensemble − mean component acc)")
    )

    (_band + _line).properties(width=600)
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

    If higher gain merely reflects weaker component models (easier to beat), the scatter
    below would show a negative slope regardless of ρ. Colour by ρ to see whether the
    gain-vs-mean-component relationship shifts across regularisation levels.
    """
    )
    return


@app.cell
def _(alt, pl, runs):
    _df = runs.select(
        pl.col("rho"),
        pl.col("comp_mean_acc"),
        pl.col("gain"),
    ).to_pandas()

    alt.Chart(_df).mark_circle(opacity=0.5, size=36).encode(
        x=alt.X(
            "comp_mean_acc:Q",
            title="Mean component acc (this combo)",
            scale=alt.Scale(zero=False),
        ),
        y=alt.Y(
            "gain:Q",
            title="Gain (ensemble − mean component acc)",
            scale=alt.Scale(zero=False),
        ),
        color=alt.Color("rho:N", title="ρ"),
        tooltip=["rho", "comp_mean_acc", "gain"],
    ).properties(width=600, height=400)
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
    **Aggregated by ρ** — mean component acc is derived by averaging `comp_mean_acc` across
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
            pl.col("mean_comp_acc").round(4).alias("mean component acc (all N models)"),
            pl.col("mean_gain")
            .round(4)
            .alias("mean gain (ensemble − mean component acc)"),
        ]
    )
    mo.ui.table(table_df.to_pandas(), selection=None)
    return


@app.cell
def _(mo):
    mo.md(
        """
    **Per combo** — `mean component acc (this combo)` is `comp_mean_acc` as logged by the
    pipeline: mean accuracy of the k component models in this specific combo.
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
                .alias("mean component acc (this combo)"),
                pl.col("gain").round(4).alias("gain vs mean component acc"),
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
        SELECT rho, avg("gain vs mean component acc"), avg("gain vs group solo mean") FROM per_run
        GROUP BY rho
        ORDER BY rho
        """
    )
    return


if __name__ == "__main__":
    app.run()
