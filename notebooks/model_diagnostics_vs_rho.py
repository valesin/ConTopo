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

    METRIC_MAP = {
        "morans_i": "metrics.morans_i",
        "weight_norms": "metrics.weight_norms_mean",
        "unit_distance_correlation": "metrics.unit_dist_cos_correlation",
    }
    return (
        METRIC_MAP,
        get_runs,
        make_run_multiselects,
        mo,
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
    diag_runs = get_runs("diagnostics")
    model_runs = get_runs("model")
    print(f"diagnostics: {len(diag_runs)}, models: {len(model_runs)}")
    return diag_runs, model_runs


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
        SELECT
            *
        FROM
            model_runs
        WHERE
            {_where}
        ORDER BY
            "params.rho"
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
        len(model_flt) == 0, mo.callout(mo.md("No runs match the filter."), kind="warn")
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
def _(METRIC_MAP, mo):
    diagnostic = mo.ui.dropdown(
        options=list(METRIC_MAP.keys()),
        value="morans_i",
        label="Diagnostic",
    )
    diagnostic
    return (diagnostic,)


@app.cell(hide_code=True)
def _(METRIC_MAP, diag_runs, diagnostic, mo, model_flt, rho_ui):
    mo.stop(not rho_ui.value, mo.callout(mo.md("Select at least one ρ."), kind="warn"))
    _metric_col = METRIC_MAP[diagnostic.value]
    _rho_in = ", ".join(f"'{r}'" for r in rho_ui.value)
    flt = mo.sql(
        f"""
        SELECT
            CAST(m."params.rho" AS DOUBLE) AS rho,
            CAST(d."{_metric_col}" AS DOUBLE) AS metric,
            CAST(m."metrics.test_accuracy" AS DOUBLE) AS accuracy
        FROM diag_runs d
        JOIN model_flt m ON d."tags.parent_run_id" = m."run_id"
        WHERE d."params.diagnostic_metric" = '{diagnostic.value}'
          AND d."params.split" = 'test'
          AND m."params.rho" IN ({_rho_in})
        """
    )
    return (flt,)


@app.cell(hide_code=True)
def _(flt, mo):
    _df = mo.sql(
        f"""
        SELECT rho, count(*) AS n
        FROM flt
        GROUP BY rho
        ORDER BY rho
        """
    )
    return


@app.cell
def _(diagnostic, flt, np, plt):
    _rho = flt["rho"].to_numpy()
    _metric = flt["metric"].to_numpy()
    _unique_rhos = np.sort(np.unique(_rho))
    _means = np.array([_metric[_rho == r].mean() for r in _unique_rhos])

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.scatter(_rho, _metric, alpha=0.5, s=15, color="steelblue", zorder=2)
    ax.plot(_unique_rhos, _means, color="firebrick", zorder=3)
    ax.set_xlabel("ρ")
    ax.set_ylabel(diagnostic.value)
    ax.set_title(f"ρ vs {diagnostic.value} (test split)")
    fig
    return


@app.cell
def _(diagnostic, flt):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    _rho = flt["rho"].to_numpy()
    _metric = flt["metric"].to_numpy()
    _acc = flt["accuracy"].to_numpy()
    _unique_rhos = np.sort(np.unique(_rho))
    _norm = plt.Normalize(_unique_rhos.min(), _unique_rhos.max())
    _cmap = cm.viridis

    fig2, ax2 = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for _r in _unique_rhos:
        _mask = _rho == _r
        ax2.scatter(
            _metric[_mask],
            _acc[_mask],
            color=_cmap(_norm(_r)),
            s=20,
            alpha=0.7,
            zorder=2,
        )
    _sm = cm.ScalarMappable(cmap=_cmap, norm=_norm)
    fig2.colorbar(_sm, ax=ax2, label="ρ", fraction=0.046, pad=0.02)
    ax2.set_xlabel(diagnostic.value)
    ax2.set_ylabel("test accuracy")
    ax2.set_title(f"{diagnostic.value} vs accuracy — coloured by ρ")
    fig2
    return np, plt


if __name__ == "__main__":
    app.run()
