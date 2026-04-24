import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys, os

    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlflow"))
    from src.config.notebook import setup_environment
    from mlflow_helpers import get_runs, varying_fields

    return get_runs, mo, setup_environment, varying_fields


@app.cell
def _(setup_environment):
    cfg, experiment = setup_environment()
    return


@app.cell
def _(get_runs):
    diag_runs = get_runs("diagnostics", diagnostic_metric="morans_i")
    model_runs = get_runs("model")
    print(f"diagnostics: {len(diag_runs)}, models: {len(model_runs)}")
    return diag_runs, model_runs


@app.cell
def _(diag_runs, varying_fields):
    varying_fields(diag_runs)
    return


@app.cell
def _(model_runs, varying_fields):
    varying_fields(model_runs)
    return


@app.cell(hide_code=True)
def _(diag_runs, mo, model_runs):
    flt = mo.sql(
        f"""
        SELECT
            CAST(m."params.rho" AS DOUBLE) AS rho,
            CAST(d."metrics.morans_i" AS DOUBLE) AS morans_i
        FROM
            diag_runs d
            JOIN model_runs m ON d."tags.parent_run_id" = m."run_id"
        WHERE
            d."params.split" = 'test'
            AND m."params.topology" = 'grid'
            AND m."params.epochs" = '200'
            AND m."params.early_stopping_method" = 'val_acc'
        """
    )
    return (flt,)


@app.cell
def _(flt, varying_fields):
    varying_fields(flt)
    return


@app.cell
def _(flt, mo):
    mo.sql(
        """
        SELECT rho, count(*) AS n
        FROM flt
        GROUP BY rho
        ORDER BY rho
        """
    )
    return


@app.cell
def _(flt):
    import altair as alt

    points = (
        alt.Chart(flt)
        .mark_point(opacity=0.5, size=15, color="steelblue")
        .encode(
            x=alt.X("rho:Q", title="ρ"),
            y=alt.Y("morans_i:Q", title="Moran's I"),
            tooltip=["rho", "morans_i"],
        )
    )

    mean_line = (
        alt.Chart(flt)
        .mark_line(color="firebrick")
        .encode(
            x="rho:Q",
            y="mean(morans_i):Q",
        )
    )

    (points + mean_line).properties(
        width=600, height=350, title="ρ vs Moran's I (test split)"
    )
    return


if __name__ == "__main__":
    app.run()
