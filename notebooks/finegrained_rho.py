import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys, os

    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlflow"))
    from src.config.notebook import setup_environment
    from mlflow_helpers import get_runs, varying_fields, get_metric_history
    import polars as pl
    import pandas as pd
    import altair as alt

    return (
        alt,
        get_metric_history,
        get_runs,
        mo,
        pd,
        pl,
        setup_environment,
        varying_fields,
    )


@app.cell
def _(mo):
    mo.md(
        """
    # Fine-grained ρ Analysis

    Visualise `train_topo_loss` trajectories across all ρ values.  Each line is
    one model run; colour encodes ρ so the full sweep is readable at a glance.
    Runs are grouped by epoch budget to separate the two training regimes.
    """
    )
    return


@app.cell
def _(setup_environment):
    cfg, experiment = setup_environment()
    return


@app.cell
def _(get_runs):
    df = get_runs("model")
    return (df,)


@app.cell
def _(mo):
    mo.md(
        """
    ## Inspection
    """
    )
    return


@app.cell
def _(df, varying_fields):
    varying_fields(df)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Loss Trajectories by ρ

    `make_topo_loss_chart(df, epochs)` loads all `train_topo_loss` histories for
    runs with the given epoch budget, then plots them coloured by ρ.
    """
    )
    return


@app.cell
def _(alt, get_metric_history, pd, pl):
    def make_topo_loss_chart(df, epochs, metric="train_topo_loss"):
        df_pl = pl.from_pandas(df) if not isinstance(df, pl.DataFrame) else df
        filtered = df_pl.filter(pl.col("params.epochs") == str(epochs))
        sorted_runs = (
            filtered.select(["run_id", "params.rho"])
            .with_columns(pl.col("params.rho").cast(pl.Float64).alias("rho_num"))
            .sort("rho_num")
        )

        frames = []
        for row in sorted_runs.iter_rows(named=True):
            h = get_metric_history(row["run_id"], metric)
            h["run_id"] = row["run_id"]
            h["rho"] = row["params.rho"]
            frames.append(h)

        histories = pd.concat(frames, ignore_index=True)
        rho_order = sorted(histories["rho"].unique(), key=float)

        return (
            alt.Chart(histories)
            .mark_line(opacity=0.75)
            .encode(
                x=alt.X(
                    "step:Q", title="Step", axis=alt.Axis(labels=False, ticks=False)
                ),
                y=alt.Y("value:Q", title=metric),
                color=alt.Color(
                    "rho:N",
                    sort=rho_order,
                    scale=alt.Scale(scheme="tableau20"),
                    legend=alt.Legend(title="ρ"),
                ),
                detail="run_id:N",
                tooltip=["rho:N", "step:Q", "value:Q"],
            )
            .properties(title=f"{metric}  —  epochs = {epochs}", width=700, height=350)
        )

    return (make_topo_loss_chart,)


@app.cell
def _(mo):
    mo.md(
        """
    ### Epochs = 30
    """
    )
    return


@app.cell
def _(df, make_topo_loss_chart):
    make_topo_loss_chart(df, 30)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ### Epochs = 60
    """
    )
    return


@app.cell
def _(df, make_topo_loss_chart):
    make_topo_loss_chart(df, 60)
    return


if __name__ == "__main__":
    app.run()
