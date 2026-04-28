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
    import numpy as np
    import matplotlib.pyplot as plt

    return (
        get_runs,
        make_run_multiselects,
        mo,
        np,
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
            "stopping": ["val_acc"],
            "epochs": ["200"],
        },
        "B": {
            "model_arch": ["FinetuneResNet34"],
            "topology": ["grid"],
            "stopping": ["val_loss"],
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
def _(mo):
    RHO_GROUPS = {
        "—": [],
        "All": None,
        "Main": ["0.0", "0.008", "0.04", "0.2", "1.0", "5.0"],
        "Fine [0.008–0.04]": (0.008, 0.04),
    }
    rho_group = mo.ui.radio(options=list(RHO_GROUPS.keys()), value="—", label="ρ group")
    rho_group
    return RHO_GROUPS, rho_group


@app.cell
def _(RHO_GROUPS, mo, model_flt, rho_group):
    mo.stop(
        len(model_flt) == 0,
        mo.callout(mo.md("No runs match the filter."), kind="warn"),
    )
    _available = sorted(model_flt["params.rho"].unique().to_list(), key=float)
    _group = RHO_GROUPS[rho_group.value]
    if _group is None:
        _default = _available
    elif isinstance(_group, list):
        _default = [r for r in _available if r in _group]
    else:
        _lo, _hi = _group
        _default = [r for r in _available if _lo <= float(r) <= _hi]
    rho_ui = mo.ui.multiselect(options=_available, value=_default, label="ρ values")
    rho_ui
    return (rho_ui,)


@app.cell(hide_code=True)
def _(mo, model_flt, rho_ui):
    mo.stop(not rho_ui.value, mo.callout(mo.md("Select at least one ρ."), kind="warn"))
    rho_counts = mo.sql(
        f"""
        SELECT "params.rho" AS rho, count(*) AS n
        FROM model_flt
        WHERE "params.rho" IN ({", ".join(f"'{r}'" for r in rho_ui.value)})
        GROUP BY rho
        ORDER BY rho
        """
    )
    return (rho_counts,)


@app.cell
def _(mo, model_flt, np, pl, plt, rho_ui):
    mo.stop(not rho_ui.value, mo.callout(mo.md("Select at least one ρ."), kind="warn"))
    _rho_vals = [
        r
        for r in sorted(model_flt["params.rho"].unique().to_list(), key=float)
        if r in rho_ui.value
    ]
    _data = [
        model_flt.filter(pl.col("params.rho") == r)["metrics.test_accuracy"]
        .cast(pl.Float64)
        .drop_nulls()
        .to_list()
        for r in _rho_vals
    ]

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.boxplot(
        _data,
        positions=range(len(_rho_vals)),
        patch_artist=True,
        widths=0.5,
        whis=[0, 100],
        flierprops=dict(visible=False),
        boxprops=dict(facecolor="steelblue", alpha=0.4),
        medianprops=dict(color="steelblue", linewidth=2),
        whiskerprops=dict(color="steelblue"),
        capprops=dict(color="steelblue"),
    )
    for _i, _d in enumerate(_data):
        ax.scatter([_i] * len(_d), _d, color="black", s=12, alpha=0.5, zorder=3)
    ax.scatter(
        range(len(_rho_vals)),
        [np.mean(_d) for _d in _data],
        color="firebrick",
        s=100,
        marker="D",
        zorder=4,
        label="mean",
    )
    ax.set_xticks(range(len(_rho_vals)))
    ax.set_xticklabels(_rho_vals, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("ρ")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Model accuracy by ρ")
    ax.legend(fontsize=8)
    fig
    return


if __name__ == "__main__":
    app.run()
