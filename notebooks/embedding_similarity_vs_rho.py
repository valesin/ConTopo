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

    return (
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
    mo.vstack([rho_ui, refresh_cache_ui])
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
    mo.stop(not rho_ui.value, mo.callout(mo.md("Select at least one ρ."), kind="warn"))

    _cache_dir = Path(get_artifact_cache_dir()) / "embedding_similarity"
    _cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_similarities(inference_run_id):
        _cache_file = _cache_dir / f"{inference_run_id}.npz"
        if _cache_file.exists() and not refresh_cache_ui.value:
            _c = np.load(_cache_file)
            return _c["within"], _c["across"]

        _df, _tensors = load_inference_artifacts(inference_run_id, split="test")
        _emb = _tensors["embeddings"].astype(float)
        _labels = _df["label"].values.astype(int)

        _emb_n = _emb / np.clip(np.linalg.norm(_emb, axis=1, keepdims=True), 1e-9, None)

        _w_sims = []
        for _c in range(10):
            _ec = _emb_n[_labels == _c]
            _gram = _ec @ _ec.T
            _w_sims.append(_gram[np.triu_indices(len(_ec), k=1)])
        _within = np.concatenate(_w_sims)

        _rng = np.random.default_rng(42)
        _i1 = _rng.integers(0, len(_labels), len(_within) * 4)
        _i2 = _rng.integers(0, len(_labels), len(_within) * 4)
        _cross = _labels[_i1] != _labels[_i2]
        _i1, _i2 = _i1[_cross][: len(_within)], _i2[_cross][: len(_within)]
        _across = (_emb_n[_i1] * _emb_n[_i2]).sum(axis=1)

        np.savez_compressed(_cache_file, within=_within, across=_across)
        return _within, _across

    _flt = flt.filter(flt["rho"].is_in(rho_ui.value))
    _rows = _flt.to_dicts()

    _rho_labels = []
    _within_by_rho = []
    _across_by_rho = []
    _mean_within = []
    _mean_across = []

    for _r in _rows:
        _within, _across = _compute_similarities(_r["inference_run_id"])
        _rho_labels.append(f"{float(_r['rho']):.2g}")
        _within_by_rho.append(_within)
        _across_by_rho.append(_across)
        _mean_within.append(float(_within.mean()))
        _mean_across.append(float(_across.mean()))

    rho_labels = _rho_labels
    within_by_rho = _within_by_rho
    across_by_rho = _across_by_rho
    mean_within = _mean_within
    mean_across = _mean_across

    mo.md(f"Computed similarities for **{len(rho_labels)} models**.")
    return across_by_rho, mean_across, mean_within, rho_labels, within_by_rho


@app.cell
def _(across_by_rho, np, plt, rho_labels, within_by_rho):
    _n = len(rho_labels)
    _pos_w = np.arange(_n) * 2.5 - 0.5
    _pos_a = np.arange(_n) * 2.5 + 0.5

    fig_violin, ax_v = plt.subplots(
        figsize=(max(8, _n * 1.8), 4), constrained_layout=True
    )

    _vp_w = ax_v.violinplot(
        within_by_rho, positions=_pos_w, widths=0.8, showmedians=True
    )
    _vp_a = ax_v.violinplot(
        across_by_rho, positions=_pos_a, widths=0.8, showmedians=True
    )

    for _pc in _vp_w["bodies"]:
        _pc.set_facecolor("steelblue")
        _pc.set_alpha(0.7)
    for _part in ("cmedians", "cbars", "cmins", "cmaxes"):
        _vp_w[_part].set_color("steelblue")

    for _pc in _vp_a["bodies"]:
        _pc.set_facecolor("darkorange")
        _pc.set_alpha(0.7)
    for _part in ("cmedians", "cbars", "cmins", "cmaxes"):
        _vp_a[_part].set_color("darkorange")

    ax_v.set_xticks(np.arange(_n) * 2.5)
    ax_v.set_xticklabels(rho_labels)
    ax_v.set_xlabel("ρ")
    ax_v.set_ylabel("cosine similarity")
    ax_v.set_title("Within-class vs across-class cosine similarity distribution")
    ax_v.legend(
        [_vp_w["bodies"][0], _vp_a["bodies"][0]], ["within-class", "across-class"]
    )
    fig_violin
    return


@app.cell
def _(mean_across, mean_within, plt, rho_labels):
    _rho_numeric = [float(r) for r in rho_labels]

    fig_means, ax_m = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax_m.plot(
        _rho_numeric,
        mean_within,
        marker="o",
        color="steelblue",
        label="within-class mean",
    )
    ax_m.plot(
        _rho_numeric,
        mean_across,
        marker="o",
        color="darkorange",
        label="across-class mean",
    )
    ax_m.fill_between(_rho_numeric, mean_within, mean_across, alpha=0.08, color="gray")
    ax_m.set_xlabel("ρ")
    ax_m.set_ylabel("mean cosine similarity")
    ax_m.set_title("Mean within- vs across-class cosine similarity vs ρ")
    ax_m.legend()
    ax_m.grid(alpha=0.3)
    fig_means
    return


if __name__ == "__main__":
    app.run()
