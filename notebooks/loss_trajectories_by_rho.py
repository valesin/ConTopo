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
        get_metric_history,
        make_run_multiselects,
        run_filter_clause,
    )
    import polars as pl
    import matplotlib.pyplot as plt

    return (
        get_metric_history,
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
    metric_ui = mo.ui.dropdown(
        options=["train_topo_loss", "train_loss", "val_loss"],
        value="train_topo_loss",
        label="Metric",
    )
    metric_ui
    return (metric_ui,)


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


@app.cell
def _():
    history_cache = {}
    return (history_cache,)


@app.cell
def _(get_metric_history, history_cache, metric_ui, mo, model_flt, pl, rho_ui):
    mo.stop(
        not rho_ui.value,
        mo.callout(mo.md("Select at least one ρ."), kind="warn"),
    )
    _df = model_flt.filter(pl.col("params.rho").is_in(rho_ui.value))
    histories_by_rho = {}
    for _row in _df.iter_rows(named=True):
        _rho = _row["params.rho"]
        _key = (_row["run_id"], metric_ui.value)
        if _key not in history_cache:
            history_cache[_key] = get_metric_history(_row["run_id"], metric_ui.value)
        _h = history_cache[_key]
        if _h.empty:
            continue
        histories_by_rho.setdefault(_rho, []).append(
            {
                "trial": _row["tags.trial"],
                "history": _h,
            }
        )
    return (histories_by_rho,)


@app.cell
def _(histories_by_rho, metric_ui, mo, plt):
    mo.stop(
        not histories_by_rho,
        mo.callout(
            mo.md("No metric history found for the selected runs."), kind="warn"
        ),
    )
    _all_steps = [
        s
        for runs in histories_by_rho.values()
        for _r in runs
        for s in _r["history"]["step"]
    ]
    _xlim = (min(_all_steps), max(_all_steps))

    _figs = []
    for _rho in sorted(histories_by_rho.keys(), key=float):
        _runs = histories_by_rho[_rho]
        fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
        for _r in _runs:
            _h = _r["history"]
            ax.plot(_h["step"], _h["value"], alpha=0.8, label=f"trial {_r['trial']}")
        ax.set_xlim(_xlim)
        ax.set_xlabel("Step")
        ax.set_ylabel(metric_ui.value)
        ax.set_title(f"ρ = {_rho}")
        _figs.append(fig)
    mo.vstack(_figs)
    return


if __name__ == "__main__":
    app.run()
