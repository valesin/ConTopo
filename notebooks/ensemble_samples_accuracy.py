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
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt

    return get_ensemble_results_for_groups, mo, np, pl, plt, setup_environment


@app.cell
def _(mo, setup_environment):
    cfg, experiment = setup_environment()
    mo.md(f"**Experiment:** `{experiment.name}`")
    return


@app.cell
def _(mo):
    groups_ui = mo.ui.text(value="samples9_mc", label="Groups")
    split_ui = mo.ui.dropdown(
        options=["test", "val", "train"], value="test", label="Split"
    )
    method_ui = mo.ui.radio(options=["soft", "hard"], value="soft", label="Vote method")
    mo.vstack([groups_ui, split_ui, method_ui])
    return groups_ui, method_ui, split_ui


@app.cell
def _(get_ensemble_results_for_groups, groups_ui, method_ui, mo, pl, split_ui):
    mo.stop(
        not groups_ui.value.strip(),
        mo.callout(mo.md("Enter a groups name."), kind="warn"),
    )

    _runs_pd = get_ensemble_results_for_groups(groups_ui.value.strip(), split_ui.value)
    _runs_pd = _runs_pd[_runs_pd["vote_method"] == method_ui.value]

    mo.stop(
        _runs_pd.empty,
        mo.callout(mo.md("No runs found for this configuration."), kind="warn"),
    )

    runs = pl.from_pandas(_runs_pd).with_columns(
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
    rho_order = agg["rho"].to_list()
    return agg, rho_order


@app.cell
def _(agg, plt, rho_order):
    _x = list(range(len(rho_order)))
    _mean_acc = agg["mean_acc"].to_list()
    _std_acc = [v or 0.0 for v in agg["std_acc"].to_list()]
    _mean_comp = agg["mean_comp_acc"].to_list()
    _lo = [m - s for m, s in zip(_mean_acc, _std_acc)]
    _hi = [m + s for m, s in zip(_mean_acc, _std_acc)]

    fig_vs, ax_vs = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax_vs.plot(
        _x,
        _mean_acc,
        marker="o",
        color="#636EFA",
        label="ensemble acc (mean over combos)",
    )
    ax_vs.fill_between(_x, _lo, _hi, alpha=0.15, color="#636EFA")
    ax_vs.plot(
        _x,
        _mean_comp,
        marker="o",
        color="#EF553B",
        linestyle="--",
        label="mean component acc",
    )
    ax_vs.set_xticks(_x)
    ax_vs.set_xticklabels(rho_order, rotation=45, ha="right")
    ax_vs.set_xlabel("ρ")
    ax_vs.set_ylabel("accuracy")
    ax_vs.set_title("Ensemble vs solo component accuracy")
    ax_vs.legend()
    ax_vs.grid(alpha=0.3)
    fig_vs
    return


@app.cell
def _(agg, plt, rho_order, runs):
    _pts = runs.sort("rho_numeric").select(["rho", "accuracy"]).to_pandas()
    _floor = round(float(_pts["accuracy"].min()) - 0.005, 3)
    _x = list(range(len(rho_order)))
    _mean_acc = agg["mean_acc"].to_list()

    fig_bar, ax_bar = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax_bar.bar(
        _x, [m - _floor for m in _mean_acc], bottom=_floor, color="#636EFA", alpha=0.55
    )
    for _i, _rho in enumerate(rho_order):
        _acc = _pts[_pts["rho"] == _rho]["accuracy"].values
        ax_bar.scatter(
            [_i] * len(_acc), _acc, color="#1a1a2e", s=35, alpha=0.75, zorder=3
        )
    ax_bar.set_xticks(_x)
    ax_bar.set_xticklabels(rho_order, rotation=45, ha="right")
    ax_bar.set_xlabel("ρ")
    ax_bar.set_ylabel("accuracy")
    ax_bar.set_ylim(_floor - 0.005, None)
    ax_bar.set_title(
        "Mean ensemble accuracy per ρ (bars = mean, dots = individual combos)"
    )
    ax_bar.grid(alpha=0.3)
    fig_bar
    return


@app.cell
def _(agg, plt, rho_order):
    _x = list(range(len(rho_order)))
    _mean_gain = agg["mean_gain"].to_list()
    _std_gain = [v or 0.0 for v in agg["std_gain"].to_list()]
    _lo = [m - s for m, s in zip(_mean_gain, _std_gain)]
    _hi = [m + s for m, s in zip(_mean_gain, _std_gain)]

    fig_gain, ax_gain = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax_gain.plot(_x, _mean_gain, marker="o", color="#32AB60", label="mean gain")
    ax_gain.fill_between(_x, _lo, _hi, alpha=0.15, color="#32AB60")
    ax_gain.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax_gain.set_xticks(_x)
    ax_gain.set_xticklabels(rho_order, rotation=45, ha="right")
    ax_gain.set_xlabel("ρ")
    ax_gain.set_ylabel("gain")
    ax_gain.set_title("Mean gain (ensemble − mean component acc)")
    ax_gain.legend()
    ax_gain.grid(alpha=0.3)
    fig_gain
    return


@app.cell
def _(np, plt, runs):
    _df = runs.select(["rho", "rho_numeric", "comp_mean_acc", "gain"]).to_pandas()
    _rhos = sorted(_df["rho"].unique(), key=lambda r: float(r))
    _colors = plt.cm.tab10(np.linspace(0, 1, min(len(_rhos), 10)))
    _rho_color = {r: _colors[i % 10] for i, r in enumerate(_rhos)}

    fig_sc, ax_sc = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for _rho in _rhos:
        _sub = _df[_df["rho"] == _rho]
        ax_sc.scatter(
            _sub["comp_mean_acc"],
            _sub["gain"],
            color=_rho_color[_rho],
            alpha=0.5,
            s=36,
            label=_rho,
        )
    ax_sc.set_xlabel("mean component acc (this combo)")
    ax_sc.set_ylabel("gain (ensemble − mean component acc)")
    ax_sc.set_title("Confounding check — gain vs component strength, coloured by ρ")
    ax_sc.legend(title="ρ", fontsize=7, ncol=2)
    ax_sc.grid(alpha=0.3)
    fig_sc
    return


@app.cell
def _(mo, runs):
    from scipy.stats import spearmanr

    _rho_vals = runs["rho_numeric"].to_list()
    _gain_vals = runs["gain"].to_list()
    _acc_vals = runs["comp_mean_acc"].to_list()
    _stat, _pval = spearmanr(_rho_vals, _gain_vals)
    _stat2, _pval2 = spearmanr(_rho_vals, _acc_vals)

    mo.vstack(
        [
            mo.callout(
                mo.md(
                    f"**Spearman ρ (rho_numeric vs gain) = {_stat:.3f}** &nbsp;|&nbsp; "
                    f"p = {_pval:.2e} &nbsp;|&nbsp; n = {len(_gain_vals)} combos"
                ),
                kind="info",
            ),
            mo.callout(
                mo.md(
                    f"**Spearman ρ (ensemble gain vs mean component accuracy) = {_stat2:.3f}** &nbsp;|&nbsp; "
                    f"p = {_pval2:.2e} &nbsp;|&nbsp; n = {len(_gain_vals)} combos"
                ),
                kind="info",
            ),
        ]
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
