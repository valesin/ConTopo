import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from src.config.notebook import setup_environment, compose_groups

    # from src.ensemble.selector import encode_groups_signature
    from src.repositories.functional_run_repository import search_runs
    import polars as pl
    import plotly.graph_objects as go

    return (setup_environment,)


@app.cell
def _(mo, setup_environment):
    cfg, experiment = setup_environment()
    mo.md(f"**Experiment:** `{experiment.name}`")
    return


if __name__ == "__main__":
    app.run()
