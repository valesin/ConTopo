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
        "mean_rsa_correlation": ("consistency", "metrics.mean_rsa_correlation"),
        "q_statistic": ("diversity", "metrics.q_statistic"),
        "disagreement": ("diversity", "metrics.disagreement"),
        "double_fault": ("diversity", "metrics.double_fault"),
        "correlation": ("diversity", "metrics.correlation"),
        "interrater_agreement": ("diversity", "metrics.interrater_agreement"),
        "iou_top_n": ("diversity", "metrics.iou_top_n"),
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
        value="mean_rsa_correlation",
        label="Diagnostic",
    )
    diagnostic
    return (diagnostic,)


@app.cell
def _(METRIC_MAP, diagnostic, get_runs):
    _kind, _metric_col = METRIC_MAP[diagnostic.value]
    ens_runs = get_runs(_kind)
    print(f"{_kind}: {len(ens_runs)} runs")
    return (ens_runs,)


@app.cell
def _(ens_runs):
    ens_runs
    return


@app.cell
def _(ens_runs, varying_fields):
    varying_fields(ens_runs)
    return


@app.cell(hide_code=True)
def _(METRIC_MAP, diagnostic, ens_runs, mo):
    _kind, _metric_col = METRIC_MAP[diagnostic.value]
    flt = mo.sql(
        f"""
        SELECT
            CAST("params.rho" AS DOUBLE) AS rho,
            CAST("{_metric_col}" AS DOUBLE) AS metric
        FROM ens_runs
        WHERE "params.split" = 'test'
          -- TODO: add AND clauses after reading varying_fields output
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
