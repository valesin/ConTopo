import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import sys, os

    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlflow"))
    from src.config.notebook import setup_environment
    from mlflow_helpers import (
        get_runs,
        load_inference_artifacts,
        make_run_multiselects,
        run_filter_clause,
        varying_fields,
    )
    import numpy as np
    import matplotlib.pyplot as plt

    CIFAR10_CLASSES = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    return (
        CIFAR10_CLASSES,
        get_runs,
        load_inference_artifacts,
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
    infer_runs = get_runs("inference", split="test")
    print(f"models: {len(model_runs)}, inference: {len(infer_runs)}")
    return infer_runs, model_runs


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
          AND "tags.trial" = '0'
        ORDER BY CAST("params.rho" AS DOUBLE)
        """
    )
    return (model_flt,)


@app.cell
def _(model_flt, varying_fields):
    varying_fields(model_flt)
    return


@app.cell(hide_code=True)
def _(infer_runs, mo, model_flt):
    mo.stop(
        len(model_flt) == 0, mo.callout(mo.md("No runs match the filter."), kind="warn")
    )
    flt = mo.sql(
        """
        SELECT
            i."run_id"                    AS inference_run_id,
            m."params.rho"                AS rho,
            COALESCE(m."tags.trial", '0') AS trial
        FROM infer_runs i
        JOIN model_flt m ON i."tags.trained_model_run_id" = m."run_id"
        ORDER BY CAST(m."params.rho" AS DOUBLE), trial
        """
    )
    return (flt,)


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
def _(RHO_GROUPS, flt, mo, rho_group):
    mo.stop(len(flt) == 0, mo.callout(mo.md("No runs match the filter."), kind="warn"))
    _available = sorted(flt["rho"].unique().to_list(), key=float)
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
def _(CIFAR10_CLASSES, flt, load_inference_artifacts, mo, np, rho_ui):
    mo.stop(not rho_ui.value, mo.callout(mo.md("Select at least one ρ."), kind="warn"))

    _flt = flt.filter(flt["rho"].is_in(rho_ui.value))
    _rows = _flt.to_dicts()
    _n_classes = len(CIFAR10_CLASSES)

    _rho_labels = []
    _acc_matrix = []

    for _r in _rows:
        _df, _ = load_inference_artifacts(_r["inference_run_id"], split="test")
        _labels = _df["label"].values.astype(int)
        _preds = _df["prediction"].values.astype(int)
        _acc_by_class = np.array(
            [
                (
                    (_preds[_labels == c] == c).mean()
                    if (_labels == c).any()
                    else float("nan")
                )
                for c in range(_n_classes)
            ]
        )
        _rho_labels.append(f"{float(_r['rho']):.2g}")
        _acc_matrix.append(_acc_by_class)

    acc_matrix = np.stack(_acc_matrix)
    rho_labels = _rho_labels

    mo.md(
        f"Loaded **{len(rho_labels)} models** — **{_n_classes} classes** × **{acc_matrix.shape[0]} ρ values**."
    )
    return acc_matrix, rho_labels


@app.cell
def _(CIFAR10_CLASSES, acc_matrix, plt, rho_labels):
    _n_rho = len(rho_labels)
    _fig_w = max(6, _n_rho * 0.6 + 2)

    fig, ax = plt.subplots(figsize=(_fig_w, 4), constrained_layout=True)
    im = ax.imshow(acc_matrix.T, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    fig.colorbar(im, ax=ax, label="accuracy", fraction=0.03, pad=0.02)
    ax.set_xticks(range(_n_rho))
    ax.set_xticklabels(rho_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(CIFAR10_CLASSES)))
    ax.set_yticklabels(CIFAR10_CLASSES, fontsize=9)
    ax.set_xlabel("ρ")
    ax.set_title("Per-class test accuracy vs ρ")
    fig
    return


if __name__ == "__main__":
    app.run()
