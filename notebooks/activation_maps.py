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
    from src.losses.topographic import get_grid_shape
    import numpy as np
    import matplotlib.pyplot as plt

    return (
        get_grid_shape,
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
def _(mo):
    topology_ui = mo.ui.multiselect(
        options=["grid", "torus"],
        value=["grid"],
        label="Topology",
    )
    topology_ui
    return (topology_ui,)


@app.cell
def _(infer_runs, mo, model_runs, topology_ui):
    mo.stop(
        not topology_ui.value,
        mo.callout(mo.md("Select at least one topology."), kind="warn"),
    )
    _topo = ", ".join(f"'{t}'" for t in topology_ui.value)
    flt = mo.sql(
        f"""
        SELECT
            i."run_id"                          AS inference_run_id,
            m."params.rho"                      AS rho,
            m."tags.trial"     AS trial
        FROM infer_runs i
        JOIN model_runs m ON i."tags.trained_model_run_id" = m."run_id"
        WHERE m."params.topology" IN ({_topo})
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
def _(flt, load_inference_artifacts, mo, rho_ui):
    from collections import Counter

    mo.stop(not rho_ui.value, mo.callout(mo.md("Select at least one ρ."), kind="warn"))
    _flt = flt.filter(flt["rho"].is_in(rho_ui.value))

    _rows = _flt.to_dicts()
    _rho_counts = Counter(r["rho"] for r in _rows)

    _loaded = {}
    _label_order = []
    for _r in _rows:
        _rho_str = f"{float(_r['rho']):.2g}"
        _lbl = f"ρ={_rho_str}" + (
            f" · t{_r['trial']}" if _rho_counts[_r["rho"]] > 1 else ""
        )
        _df, _tensors = load_inference_artifacts(_r["inference_run_id"], split="test")
        _loaded[_r["inference_run_id"]] = {
            "label": _lbl,
            "embeddings": _tensors["embeddings"],
        }
        _label_order.append(_lbl)

    all_data = _loaded
    _first = next(iter(all_data.values()))
    n_samples = _first["embeddings"].shape[0]
    emb_dim = _first["embeddings"].shape[1]
    label_order = _label_order

    mo.md(
        f"Loaded **{len(all_data)} models** — "
        f"**{n_samples:,} samples** × **{emb_dim} dims** each."
    )
    return all_data, emb_dim, label_order, n_samples


@app.cell
def _(mo, n_samples):
    img_idx = mo.ui.slider(
        0, n_samples - 1, value=0, label="Image index", show_value=True
    )
    img_idx
    return (img_idx,)


@app.cell
def _(all_data, emb_dim, get_grid_shape, img_idx, label_order, np, plt):
    _h, _w = get_grid_shape(emb_dim)
    _embs = {v["label"]: v["embeddings"][img_idx.value] for v in all_data.values()}
    _all_vals = np.concatenate(list(_embs.values()))
    _g_min, _g_max = float(_all_vals.min()), float(_all_vals.max())
    _n = len(label_order)

    def _make_fig(norm: str, title: str, vmin: float, vmax: float):
        fig, axes = plt.subplots(
            1, _n, figsize=(2 * _n + 1, 2.5), constrained_layout=True
        )
        if _n == 1:
            axes = [axes]
        for ax, lbl in zip(axes, label_order):
            g = _embs[lbl].reshape(_h, _w).astype(float)
            if norm == "sym":
                lo, hi = g.min(), g.max()
                g = 2 * (g - lo) / (hi - lo + 1e-9) - 1
            im = ax.imshow(g, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
            ax.set_title(lbl, fontsize=9, fontfamily="monospace")
            ax.axis("off")
        fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02, format="%.2f")
        fig.suptitle(title, fontsize=11, x=0.02, ha="left", color="#555")
        return fig

    per_model_fig = _make_fig("sym", "Per-model scale  (min → −1, max → +1)", -1.0, 1.0)
    shared_fig = _make_fig(
        "global", f"Shared scale  [{_g_min:.3f}, {_g_max:.3f}]", _g_min, _g_max
    )
    return per_model_fig, shared_fig


@app.cell
def _(per_model_fig):
    per_model_fig
    return


@app.cell
def _(shared_fig):
    shared_fig
    return


if __name__ == "__main__":
    app.run()
