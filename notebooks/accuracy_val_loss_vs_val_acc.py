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

    return get_base_model_list, pl, plt, setup_environment, varying_fields


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
def _(df, pl):
    bare_df = df.filter(pl.col("params.epochs").cast(pl.Int32) == 200).select(
        [
            "params.rho",
            "params.early_stopping_method",
            "metrics.test_accuracy",
        ]
    )
    return (bare_df,)


@app.cell
def _(bare_df, pl):
    agg = (
        bare_df.group_by(["params.rho", "params.early_stopping_method"])
        .agg(
            [
                pl.col("metrics.test_accuracy").mean().alias("acc_mean"),
                pl.col("metrics.test_accuracy").std().alias("acc_std"),
                pl.len().alias("n"),
            ]
        )
        .sort(["params.rho", "params.early_stopping_method"])
    )
    return (agg,)


@app.cell
def _(agg, plt):
    pdf = agg.to_pandas().rename(
        columns={
            "params.rho": "rho",
            "params.early_stopping_method": "early_stopping_method",
        }
    )
    pdf["acc_std"] = pdf["acc_std"].fillna(0.0)

    rho_order = sorted(pdf["rho"].unique(), key=lambda v: float(v))
    rho_pos = {r: i for i, r in enumerate(rho_order)}

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, g in pdf.groupby("early_stopping_method"):
        g = g.sort_values("rho", key=lambda s: s.map(lambda v: float(v)))
        x = [rho_pos[r] for r in g["rho"]]
        y = g["acc_mean"]
        err = g["acc_std"]
        ax.plot(x, y, marker="o", label=str(method))
        ax.fill_between(x, y - err, y + err, alpha=0.2)

    ax.set_xticks(range(len(rho_order)))
    ax.set_xticklabels(rho_order)
    ax.set_xlabel("ρ")
    ax.set_ylabel("train accuracy")
    ax.legend(title="early stopping method")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()
