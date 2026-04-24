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
    df = get_runs("model")
    print(f"Found {len(df)} trained models")
    print(df.columns)
    return (df,)


@app.cell
def _(df, varying_fields):
    varying_fields(df)
    return


@app.cell(hide_code=True)
def _(df, mo):
    flt = mo.sql(
        f"""
        SELECT rho, trial, acc
        FROM (
            SELECT
                "params.rho" AS rho,
                "tags.trial" AS trial,
                "metrics.test_accuracy" AS acc,
                COUNT(*) OVER (PARTITION BY "params.rho") AS n
            FROM df
            WHERE "params.epochs" = '200'
              AND "params.early_stopping_method" = 'val_acc'
              AND "params.topology" = 'grid'
        )
        WHERE n >= 10
        """
    )
    return (flt,)


@app.cell
def _(flt):
    flt
    return


@app.cell
def _(flt, mo):
    _df = mo.sql(
        f"""
        SELECT rho, count(*) FROM flt
        GROUP BY rho
        ORDER BY rho
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We have 10 additional runs for the first values
    """
    )
    return


@app.cell
def _(flt):
    import altair as alt

    # 1. Base configuration
    base = alt.Chart(flt).encode(x=alt.X("rho:O", title="Rho Bins"))

    # 2. Boxplot layer
    boxplot = base.mark_boxplot(extent="min-max", outliers=False).encode(
        y=alt.Y("acc:Q", scale=alt.Scale(zero=False)),
        color=alt.Color("rho:N", legend=None),
    )

    # 3. Average diamonds
    averages = base.mark_point(
        color="firebrick", size=100, shape="diamond", filled=True
    ).encode(y="mean(acc):Q")

    # 4. Individual data points (Smaller diameter and Black)
    points = base.mark_point(
        opacity=0.5, size=10, color="black"  # Reduced diameter  # Changed to black
    ).encode(y=alt.Y("acc:Q"))

    # 5. Final Layering (Points on top)
    chart = (boxplot + averages + points).properties(
        width=500, height=350, title="Model Accuracy by Rho"
    )

    chart
    return


if __name__ == "__main__":
    app.run()
