import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import sys, os
    from pathlib import Path

    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlflow"))
    from src.config.notebook import setup_environment
    from mlflow_helpers import (
        get_artifact_cache_dir,
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
        Path,
        get_artifact_cache_dir,
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
    refresh_cache_ui = mo.ui.checkbox(value=False, label="Refresh cache")
    mo.vstack(
        [
            mo.callout(
                mo.md(
                    "t-SNE fits ~10k samples per model — expect 30–90 s per ρ value."
                ),
                kind="info",
            ),
            rho_ui,
            refresh_cache_ui,
        ]
    )
    return refresh_cache_ui, rho_ui


@app.cell
def _(
    Path,
    flt,
    get_artifact_cache_dir,
    load_inference_artifacts,
    mo,
    np,
    refresh_cache_ui,
    rho_ui,
):
    from sklearn.manifold import TSNE

    mo.stop(not rho_ui.value, mo.callout(mo.md("Select at least one ρ."), kind="warn"))

    _n_components = 2
    _perplexity = 30
    _random_state = 42

    _cache_dir = Path(get_artifact_cache_dir()) / "embedding_tsne"
    _cache_dir.mkdir(parents=True, exist_ok=True)

    def _fit_or_load_tsne(inference_run_id):
        _cache_file = (
            _cache_dir
            / f"{inference_run_id}_nc{_n_components}_p{_perplexity}_rs{_random_state}.npz"
        )
        if _cache_file.exists() and not refresh_cache_ui.value:
            _c = np.load(_cache_file)
            return _c["coords"], _c["labels"]

        _df, _tensors = load_inference_artifacts(inference_run_id, split="test")
        _emb = _tensors["embeddings"].astype(float)
        _labels = _df["label"].values.astype(int)
        _coords = TSNE(
            n_components=_n_components,
            perplexity=_perplexity,
            random_state=_random_state,
        ).fit_transform(_emb)

        np.savez_compressed(_cache_file, coords=_coords, labels=_labels)
        return _coords, _labels

    _flt = flt.filter(flt["rho"].is_in(rho_ui.value))
    _rows = _flt.to_dicts()

    tsne_results = {}
    for _r in _rows:
        _coords, _labels = _fit_or_load_tsne(_r["inference_run_id"])
        _lbl = f"ρ={float(_r['rho']):.2g}"
        tsne_results[_lbl] = (_coords, _labels)

    mo.md(f"Fitted t-SNE for **{len(tsne_results)} models**.")
    return (tsne_results,)


@app.cell
def _(CIFAR10_CLASSES, np, plt, tsne_results):
    _n = len(tsne_results)
    _colors = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, axes = plt.subplots(1, _n, figsize=(5 * _n, 5), constrained_layout=True)
    if _n == 1:
        axes = [axes]

    for ax, (rho_label, (coords, labels)) in zip(axes, tsne_results.items()):
        for c, (cls_name, col) in enumerate(zip(CIFAR10_CLASSES, _colors)):
            mask = labels == c
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=[col],
                s=2,
                alpha=0.4,
                rasterized=True,
            )
        ax.set_title(rho_label, fontsize=13)
        ax.axis("off")

    _handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=_colors[c], markersize=7
        )
        for c in range(10)
    ]
    axes[-1].legend(
        _handles,
        CIFAR10_CLASSES,
        fontsize=8,
        markerscale=1,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    fig.suptitle("t-SNE of embeddings — CIFAR-10 test set", fontsize=14)
    fig
    return


if __name__ == "__main__":
    app.run()
