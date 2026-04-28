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
        get_metric_history,
    )
    import numpy as np
    import matplotlib.pyplot as plt

    return (
        get_metric_history,
        get_runs,
        make_run_multiselects,
        mo,
        np,
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
def _(get_metric_history, mo, model_flt, plt):
    mo.stop(
        len(model_flt) == 0, mo.callout(mo.md("No runs match the filter."), kind="warn")
    )

    _metric = "train_topo_loss"
    _rows = model_flt.select(["run_id", "params.rho"]).to_dicts()

    _rho_vals = sorted({float(r["params.rho"]) for r in _rows})
    _rho_rank = {v: i for i, v in enumerate(_rho_vals)}
    _cmap = plt.cm.viridis
    _norm = plt.Normalize(0, max(len(_rho_vals) - 1, 1))

    _frames = {}
    for _r in _rows:
        _rho = float(_r["params.rho"])
        _h = get_metric_history(_r["run_id"], _metric)
        _frames.setdefault(_rho, []).append((_h["step"].values, _h["value"].values))

    _all_steps = [
        s for _runs in _frames.values() for _steps, _ in _runs for s in _steps
    ]
    _xlim = (min(_all_steps), max(_all_steps)) if _all_steps else (0, 1)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for _rho in sorted(_frames):
        _col = _cmap(_norm(_rho_rank[_rho]))
        for _steps, _vals in _frames[_rho]:
            ax.plot(_steps, _vals, color=_col, alpha=0.7, linewidth=0.9)

    _sm = plt.cm.ScalarMappable(cmap=_cmap, norm=_norm)
    _sm.set_array([])
    _cbar = fig.colorbar(_sm, ax=ax)
    _cbar.set_ticks(range(len(_rho_vals)))
    _cbar.set_ticklabels([f"{v:.2g}" for v in sorted(_rho_vals)])
    _cbar.set_label("ρ")

    ax.set_xlim(_xlim)
    ax.set_xlabel("step")
    ax.set_ylabel(_metric)
    ax.set_title(f"{_metric} trajectories coloured by ρ")
    ax.grid(alpha=0.3)
    fig
    return


if __name__ == "__main__":
    app.run()
