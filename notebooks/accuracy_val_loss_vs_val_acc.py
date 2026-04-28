import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys, os

    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlflow"))
    from src.config.notebook import setup_environment
    from mlflow_helpers import (
        get_runs,
        varying_fields,
        make_run_multiselects,
        run_filter_clause,
    )
    import polars as pl
    import matplotlib.pyplot as plt

    return (
        get_runs,
        make_run_multiselects,
        mo,
        pl,
        plt,
        run_filter_clause,
        setup_environment,
        varying_fields,
    )


@app.cell
def _(setup_environment):
    cfg, experiment = setup_environment()
    return


@app.cell
def _(get_runs):
    model_runs = get_runs("model")
    print(f"models: {len(model_runs)}")
    return (model_runs,)


@app.cell
def _(model_runs, varying_fields):
    varying_fields(model_runs)
    return


@app.cell
def _(mo):
    FIELDS = {
        "model_arch": (
            "params.model_arch",
            "Model arch",
            ["LinearResNet18", "FinetuneResNet34"],
        ),
        "topology": ("params.topology", "Topology", ["grid", "torus"]),
        "stopping": (
            "params.early_stopping_method",
            "Early stopping",
            ["val_acc", "val_loss"],
        ),
        "epochs": ("params.epochs", "Epochs", ["200", "100"]),
    }
    PRESETS = {
        "A": {
            "model_arch": ["LinearResNet18"],
            "topology": ["grid"],
            "stopping": ["val_acc", "val_loss"],
            "epochs": ["200"],
        },
        "B": {
            "model_arch": ["FinetuneResNet34"],
            "topology": ["grid"],
            "stopping": ["val_acc", "val_loss"],
            "epochs": ["100"],
        },
    }
    preset = mo.ui.radio(options=list(PRESETS.keys()), value="A", label="Preset")
    preset
    return FIELDS, PRESETS, preset


@app.cell
def _(FIELDS, PRESETS, make_run_multiselects, mo, preset):
    controls = make_run_multiselects(mo, FIELDS, PRESETS[preset.value])
    mo.vstack(list(controls.values()))
    return (controls,)


@app.cell(hide_code=True)
def _(FIELDS, controls, mo, model_runs, run_filter_clause):
    _where = run_filter_clause(mo, FIELDS, controls)
    model_flt = mo.sql(
        f"""
        SELECT * FROM model_runs
        WHERE {_where}
        ORDER BY CAST("params.rho" AS DOUBLE)
        """
    )
    return (model_flt,)


@app.cell
def _(model_flt, varying_fields):
    varying_fields(model_flt)
    return


@app.cell
def _(mo, model_flt, pl):
    mo.stop(
        len(model_flt) == 0, mo.callout(mo.md("No runs match the filter."), kind="warn")
    )
    agg = (
        model_flt.select(
            ["params.rho", "params.early_stopping_method", "metrics.test_accuracy"]
        )
        .with_columns(pl.col("metrics.test_accuracy").cast(pl.Float64))
        .drop_nulls("metrics.test_accuracy")
        .group_by(["params.rho", "params.early_stopping_method"])
        .agg(
            [
                pl.col("metrics.test_accuracy").mean().alias("acc_mean"),
                pl.col("metrics.test_accuracy").std().alias("acc_std"),
                pl.len().alias("n"),
            ]
        )
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

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
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
    ax.set_ylabel("test accuracy")
    ax.legend(title="early stopping method")
    ax.grid(alpha=0.3)
    fig
    return


if __name__ == "__main__":
    app.run()
