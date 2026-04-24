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

    METRIC_MAP = {
        "morans_i": "metrics.morans_i",
        "weight_norms": "metrics.weight_norms_mean",
        "unit_distance_correlation": "metrics.unit_dist_cos_correlation",
    }
    return METRIC_MAP, get_runs, mo, setup_environment, varying_fields


@app.cell
def _(setup_environment):
    cfg, experiment = setup_environment()
    return


@app.cell
def _(METRIC_MAP, mo):
    diagnostic = mo.ui.dropdown(
        options=list(METRIC_MAP.keys()),
        value="morans_i",
        label="Diagnostic",
    )
    diagnostic
    return (diagnostic,)


@app.cell
def _(diagnostic, get_runs):
    diag_runs = get_runs("diagnostics", diagnostic_metric=diagnostic.value)
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
def _(METRIC_MAP, diag_runs, diagnostic, mo, model_runs):
    _metric_col = METRIC_MAP[diagnostic.value]
    flt = mo.sql(
        f"""
        SELECT
            CAST(m."params.rho" AS DOUBLE) AS rho,
            CAST(d."{_metric_col}" AS DOUBLE) AS metric
        FROM diag_runs d
        JOIN model_runs m ON d."tags.parent_run_id" = m."run_id"
        WHERE d."params.split" = 'test'
          AND m."params.topology" = 'grid'
          AND m."params.epochs" = '200'
          AND m."params.early_stopping_method" = 'val_acc'
        """
    )
    return (flt,)


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
def _(diagnostic, flt):
    import altair as alt

    points = (
        alt.Chart(flt)
        .mark_point(opacity=0.5, size=15, color="steelblue")
        .encode(
            x=alt.X("rho:Q", title="ρ"),
            y=alt.Y("metric:Q", title=diagnostic.value),
            tooltip=["rho", "metric"],
        )
    )

    mean_line = (
        alt.Chart(flt)
        .mark_line(color="firebrick")
        .encode(
            x="rho:Q",
            y="mean(metric):Q",
        )
    )

    (points + mean_line).properties(
        width=600, height=350, title=f"ρ vs {diagnostic.value} (test split)"
    )
    return


if __name__ == "__main__":
    app.run()
