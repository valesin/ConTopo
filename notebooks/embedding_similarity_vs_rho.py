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
        get_runs,
        load_inference_artifacts,
        varying_fields,
        get_artifact_cache_dir,
    )
    import numpy as np
    import matplotlib.pyplot as plt

    return (
        Path,
        get_artifact_cache_dir,
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

        # L2-normalise
        _emb_n = _emb / np.clip(np.linalg.norm(_emb, axis=1, keepdims=True), 1e-9, None)

        # Within-class: all pairs per class via gram matrix, upper triangle
        _w_sims = []
        for _c in range(10):
            _ec = _emb_n[_labels == _c]
            _gram = _ec @ _ec.T
            _w_sims.append(_gram[np.triu_indices(len(_ec), k=1)])
        _within = np.concatenate(_w_sims)

        # Across-class: random pairs matched to within-class count
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
        [_vp_w["bodies"][0], _vp_a["bodies"][0]],
        ["within-class", "across-class"],
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
