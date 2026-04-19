import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys, os

    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlflow"))
    from src.config.notebook import setup_environment
    from mlflow_helpers import get_base_model_list, varying_fields
    import polars as pl
    import pandas as pd
    import matplotlib.pyplot as plt

    return get_base_model_list, mo, pl, setup_environment, varying_fields


@app.cell
def _(setup_environment):
    cfg, experiment = setup_environment()
    (cfg, experiment)
    return (experiment,)


@app.cell
def _(experiment, get_base_model_list):
    df = get_base_model_list(experiment)
    print(len(df))
    df.columns
    return (df,)


@app.cell
def _(df, varying_fields):
    varying_fields(df)
    return


@app.cell
def _(df, pl, varying_fields):
    filtered = df.filter(pl.col("params.epochs") == str(30))
    varying_fields(filtered)
    return (filtered,)


@app.cell
def _():
    return


@app.cell
def _(filtered, mo):
    slider = mo.ui.slider(0, len(filtered["params.rho"]))
    mo.md(f"Choose a value: {slider}")
    return


app._unparsable_cell(
    r"""
    import altair as
    # Understand how to plot in a reactive way
    # Understand how to retrieve the loss path from mlflow
    """,
    name="_",
)


if __name__ == "__main__":
    app.run()
