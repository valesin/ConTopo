import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import sys, os

    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlflow"))
    from src.config.notebook import setup_environment
    from mlflow_helpers import get_runs, load_inference_artifacts, varying_fields
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
        mo,
        np,
        plt,
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
def _(infer_runs, mo, model_runs):
    flt = mo.sql(
        f"""
        SELECT
            i."run_id"                       AS inference_run_id,
            m."params.rho"                   AS rho,
            COALESCE(m."tags.trial", '0')    AS trial
        FROM infer_runs i
        JOIN model_runs m ON i."tags.trained_model_run_id" = m."run_id"
        WHERE m."params.topology" = 'grid'
          AND m."params.epochs" = '200'
          AND m."params.early_stopping_method" = 'val_acc'
          AND m."params.model_arch" = 'LinearResNet18'
          AND m."tags.trial" = '0'
        ORDER BY CAST(m."params.rho" AS DOUBLE), trial
        """
    )
    return (flt,)


@app.cell
def _(flt, mo):
    mo.stop(len(flt) == 0, mo.callout(mo.md("No runs match the filter."), kind="warn"))
    _rhos = sorted(flt["rho"].unique().to_list(), key=float)
    rho_ui = mo.ui.multiselect(options=_rhos, value=_rhos, label="ρ values")
    rho_ui
    return (rho_ui,)


@app.cell
def _(CIFAR10_CLASSES, flt, load_inference_artifacts, mo, np, rho_ui):
    mo.stop(not rho_ui.value, mo.callout(mo.md("Select at least one ρ."), kind="warn"))

    _flt = flt.filter(flt["rho"].is_in(rho_ui.value))
    _rows = _flt.to_dicts()
    _n_classes = len(CIFAR10_CLASSES)

    _rho_labels = []
    _acc_matrix = []  # list of per-class accuracy arrays, one per run

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

    acc_matrix = np.stack(_acc_matrix)  # shape: (n_rho, n_classes)
    rho_labels = _rho_labels

    mo.md(
        f"Loaded **{len(rho_labels)} models** — "
        f"**{_n_classes} classes** × **{acc_matrix.shape[0]} ρ values**."
    )
    return acc_matrix, rho_labels


@app.cell
def _(CIFAR10_CLASSES, acc_matrix, plt, rho_labels):
    _n_rho = len(rho_labels)
    _fig_w = max(6, _n_rho * 0.6 + 2)

    fig, ax = plt.subplots(figsize=(_fig_w, 4), constrained_layout=True)
    im = ax.imshow(
        acc_matrix.T,  # shape: (n_classes, n_rho)
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
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
