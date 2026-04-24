import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlflow"))
    from src.config.notebook import setup_environment
    from mlflow_helpers import get_runs, load_inference_artifacts, varying_fields
    from src.losses.topographic import get_grid_shape
    import numpy as np
    import pandas as pd
    import altair as alt

    return (
        alt,
        get_grid_shape,
        get_runs,
        load_inference_artifacts,
        mo,
        np,
        pd,
        setup_environment,
        varying_fields,
    )


@app.cell
def _(mo):
    mo.md(
        """
    # Topographic Activation Maps

    Visualises how a single image activates the embedding grid of models trained
    with different topographic regularisation strengths (ρ).

    Each heatmap is the pre-ReLU embedding vector (shape `[emb_dim]`) reshaped to
    the 2-D grid used by the topographic loss (`get_grid_shape(emb_dim)`), displayed
    under three normalisation schemes so that different aspects of the activation
    structure are visible simultaneously.
    """
    )
    return


@app.cell
def _(setup_environment):
    cfg, experiment = setup_environment()
    return


@app.cell
def _(mo):
    split_ui = mo.ui.dropdown(options=["test", "val"], value="test", label="Split")
    topology_ui = mo.ui.dropdown(
        options=["grid", "torus"], value="grid", label="Topology"
    )
    epochs_ui = mo.ui.dropdown(
        options=["200", "60", "30", "1"], value="200", label="Epochs"
    )
    max_trials_ui = mo.ui.number(start=1, stop=20, value=3, label="Max trials per ρ")
    mo.hstack([split_ui, topology_ui, epochs_ui, max_trials_ui], gap=2)
    return epochs_ui, max_trials_ui, split_ui, topology_ui


@app.cell
def _(epochs_ui, get_runs, pd, split_ui, topology_ui):
    from collections import Counter

    model_runs = get_runs("model", topology=topology_ui.value, epochs=epochs_ui.value)
    _infer_runs = get_runs("inference", split=split_ui.value)

    # Index inference runs by model run_id; keep the first finished run per model
    _infer_idx = (
        _infer_runs.dropna(subset=["tags.trained_model_run_id"])
        .drop_duplicates(subset=["tags.trained_model_run_id"])
        .set_index("tags.trained_model_run_id")
    )

    _has_trial = "params.trial" in model_runs.columns
    _pairs = []
    for _, _mrow in model_runs.iterrows():
        _mid = _mrow["run_id"]
        if _mid not in _infer_idx.index:
            continue
        _irow = _infer_idx.loc[_mid]
        _t = _mrow["params.trial"] if _has_trial else None
        _trial = str(int(float(_t))) if (_t is not None and pd.notna(_t)) else "0"
        _pairs.append(
            {
                "rho": _mrow["params.rho"],
                "rho_numeric": float(_mrow["params.rho"]),
                "trial": _trial,
                "model_run_id": _mid,
                "inference_run_id": _irow["run_id"],
            }
        )

    _pairs = sorted(_pairs, key=lambda x: (x["rho_numeric"], x["trial"]))

    # Add display labels; include trial suffix only when a rho has multiple entries
    _rho_counts = Counter(p["rho"] for p in _pairs)
    for _p in _pairs:
        if _rho_counts[_p["rho"]] > 1:
            _p["label"] = f"ρ={_p['rho']} · t{_p['trial']}"
        else:
            _p["label"] = f"ρ={_p['rho']}"

    rho_infer = _pairs
    return model_runs, rho_infer


@app.cell
def _(mo, model_runs, rho_infer, varying_fields):
    mo.stop(
        not rho_infer,
        mo.callout(
            mo.md(
                "No (model → inference) pairs found for the selected filters. "
                "Adjust topology/epochs or run the inference stage first."
            ),
            kind="warn",
        ),
    )
    mo.vstack(
        [
            mo.md(
                f"Found **{len(rho_infer)} model–inference pairs**. Varying fields in this model selection:"
            ),
            mo.ui.table(varying_fields(model_runs)),
        ]
    )
    return


@app.cell
def _(max_trials_ui, mo, rho_infer):
    from collections import defaultdict as _defaultdict

    _seen = _defaultdict(int)
    _capped = []
    for _item in rho_infer:
        if _seen[_item["rho"]] < max_trials_ui.value:
            _capped.append(_item)
            _seen[_item["rho"]] += 1

    rho_infer_sel = _capped

    mo.stop(
        not rho_infer_sel,
        mo.callout(
            mo.md("No pairs after applying trial cap. Increase max trials."),
            kind="warn",
        ),
    )
    return (rho_infer_sel,)


@app.cell
def _(mo, rho_infer_sel):
    _options = {item["label"]: item["model_run_id"] for item in rho_infer_sel}
    model_select = mo.ui.multiselect(
        options=_options,
        value=list(_options.keys()),
        label="Models to display",
    )
    model_select
    return (model_select,)


@app.cell
def _(load_inference_artifacts, mo, model_select, rho_infer_sel, split_ui):
    mo.stop(
        not model_select.value,
        mo.callout(mo.md("Select at least one model above."), kind="warn"),
    )

    _selected = set(model_select.value)
    _lookup = {item["model_run_id"]: item for item in rho_infer_sel}
    _loaded = {}
    for _run_id in model_select.value:
        _item = _lookup[_run_id]
        _df, _tensors = load_inference_artifacts(
            _item["inference_run_id"], split=split_ui.value
        )
        _loaded[_run_id] = {
            "label": _item["label"],
            "embeddings": _tensors["embeddings"],  # [N, emb_dim] float32
            "df": _df,
        }

    all_data = _loaded
    n_samples = next(iter(all_data.values()))["embeddings"].shape[0]
    emb_dim = next(iter(all_data.values()))["embeddings"].shape[1]

    mo.md(
        f"Loaded embeddings for **{len(all_data)} models** — "
        f"**{n_samples:,} samples** × **{emb_dim} dims** each."
    )
    return all_data, emb_dim, n_samples


@app.cell
def _(mo, n_samples):
    img_idx = mo.ui.slider(
        0, n_samples - 1, value=0, label="Image index", show_value=True
    )
    return (img_idx,)


@app.cell
def _(
    all_data,
    alt,
    emb_dim,
    get_grid_shape,
    img_idx,
    mo,
    np,
    pd,
    rho_infer_sel,
):
    _h, _w = get_grid_shape(emb_dim)

    # Preserve sorted order from rho_infer_sel, restricted to loaded models
    _loaded_ids = set(all_data.keys())
    _order = [
        item["model_run_id"]
        for item in rho_infer_sel
        if item["model_run_id"] in _loaded_ids
    ]

    _emb_by_id = {mid: all_data[mid]["embeddings"][img_idx.value] for mid in _order}

    # Pull ground-truth label and prediction from the first model's result table.
    # All runs share the same test-set ordering so row index == image index.
    _first_df = next(iter(all_data.values()))["df"]
    _label_col = next(
        (c for c in _first_df.columns if c.lower() in ("label", "labels")), None
    )
    _pred_col = next(
        (c for c in _first_df.columns if c.lower() in ("pred", "preds")), None
    )
    _info = f"image **{img_idx.value}**"
    if _label_col:
        _lv = int(_first_df.iloc[img_idx.value][_label_col])
        _info += f"  |  label = **{_lv}**"
        if _pred_col:
            _pv = int(_first_df.iloc[img_idx.value][_pred_col])
            _tick = "✓" if _lv == _pv else "✗"
            _info += f"  |  pred = **{_pv}** {_tick}"

    # Global min/max across all models — used for the third normalisation row
    _all_vals = np.concatenate(list(_emb_by_id.values()))
    _g_min, _g_max = float(_all_vals.min()), float(_all_vals.max())

    def _build_df(norm: str) -> pd.DataFrame:
        rows = []
        for mid in _order:
            lbl = all_data[mid]["label"]
            g = _emb_by_id[mid].reshape(_h, _w).astype(float)
            if norm == "sym":
                lo, hi = g.min(), g.max()
                g = 2 * (g - lo) / (hi - lo + 1e-9) - 1
            for i in range(_h):
                for j in range(_w):
                    rows.append({"model": lbl, "row": i, "col": j, "v": g[i, j]})
        return pd.DataFrame(rows)

    _col_sort = [all_data[mid]["label"] for mid in _order]
    _axis_cfg = alt.Axis(labels=False, ticks=False, title=None, domain=False)

    def _faceted_heatmap(
        norm: str, title: str, scheme: str, domain: list
    ) -> alt.FacetChart:
        return (
            alt.Chart(_build_df(norm))
            .mark_rect()
            .encode(
                x=alt.X("col:O", axis=_axis_cfg),
                y=alt.Y("row:O", axis=_axis_cfg),
                color=alt.Color(
                    "v:Q",
                    scale=alt.Scale(scheme=scheme, domain=domain),
                    legend=alt.Legend(
                        title=None,
                        orient="right",
                        gradientLength=100,
                        format=".2f",
                    ),
                ),
                tooltip=[
                    "model:N",
                    alt.Tooltip("row:O", title="grid row"),
                    alt.Tooltip("col:O", title="grid col"),
                    alt.Tooltip("v:Q", format=".4f", title="value"),
                ],
            )
            .properties(width=130, height=130)
            .facet(
                column=alt.Column(
                    "model:N",
                    sort=_col_sort,
                    header=alt.Header(
                        title=None,
                        labelFontSize=12,
                        labelFont="monospace",
                    ),
                ),
                title=alt.TitleParams(title, anchor="start", fontSize=12, color="#555"),
            )
        )

    _chart = (
        alt.vconcat(
            _faceted_heatmap(
                "sym",
                "Per-model scale  (min → −1, max → +1)",
                "redblue",
                [-1.0, 1.0],
            ),
            _faceted_heatmap(
                "global",
                f"Shared scale  (range [{_g_min:.3f}, {_g_max:.3f}] across all models)",
                "redblue",
                [_g_min, _g_max],
            ),
            spacing=28,
        )
        .resolve_scale(
            color="independent",
        )
        .properties(
            title=alt.TitleParams(
                f"Topographic activation map — {_info}  |  grid {_h}×{_w}  (emb_dim={emb_dim})",
                anchor="start",
                fontSize=14,
            )
        )
        .configure_view(
            strokeWidth=0,
        )
        .configure_concat(
            spacing=28,
        )
    )

    mo.vstack(
        [
            img_idx,
            # mo.callout(mo.md(f"Showing {_info}"), kind="info"),
            _chart,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
