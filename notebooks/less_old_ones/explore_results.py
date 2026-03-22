import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

with app.setup:
    """Imports and MLflow setup."""
    import marimo as mo
    import mlflow
    import torch
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    import os

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 120,
            "savefig.dpi": 200,
        }
    )

    # ── MLflow connection ──
    TRACKING_URI = "sqlite:///mlflow.db"
    ARTIFACTS_ROOT = "artifacts"
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.get_experiment_by_name("contopo")


@app.cell
def header():
    mo.md(f"""
    # ConTopo — Exploratory Results Analysis

    **MLflow tracking URI**: `{TRACKING_URI}`
    **Experiment**: `{experiment.name}` (id: `{experiment.experiment_id}`)
    """)
    return


@app.cell
def query_model_runs():
    """Query all finished model runs from MLflow."""
    _runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.kind = 'model' and attributes.status = 'FINISHED'",
    )

    model_df = _runs[
        ["run_id", "tags.rho", "tags.trial", "tags.topology", "tags.cfg_hash"]
    ].copy()
    model_df.columns = ["run_id", "rho", "trial", "topology", "cfg_hash"]
    model_df["rho"] = model_df["rho"].astype(float)
    model_df["trial"] = model_df["trial"].astype(int)

    mo.md(f"""
    ## 1. Model Runs Overview

    Found **{len(model_df)} model runs** across
    **{model_df['rho'].nunique()} rho values** ×
    **{model_df['trial'].nunique()} trials** ×
    **{model_df['topology'].nunique()} topologies**.
    """)
    return (model_df,)


@app.cell
def query_inference_runs():
    """Query inference runs and join with model metadata."""
    _runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.kind = 'inference' and attributes.status = 'FINISHED'",
    )

    inf_df = _runs[
        ["run_id", "tags.parent_run_id", "tags.rho", "tags.trial", "tags.topology"]
    ].copy()
    inf_df.columns = ["inf_run_id", "model_run_id", "rho", "trial", "topology"]
    inf_df["rho"] = inf_df["rho"].astype(float)
    inf_df["trial"] = inf_df["trial"].astype(int)

    _acc_col = [c for c in _runs.columns if "accuracy" in c.lower()]
    if _acc_col:
        inf_df["test_accuracy"] = _runs[_acc_col[0]].values

    mo.md(f"""
    ## 2. Inference Runs

    Found **{len(inf_df)} inference runs** with test accuracy data.

    {inf_df[["rho", "trial", "topology", "test_accuracy"]].sort_values(["rho", "trial"]).to_markdown(index=False)}
    """)
    return (inf_df,)


@app.cell
def plot_accuracy_vs_rho(inf_df):
    """Plot: Average test accuracy vs rho, with individual trial points."""
    _fig, _ax = plt.subplots(figsize=(8, 5))

    _grouped = inf_df.groupby("rho")["test_accuracy"]
    _rho_vals = sorted(inf_df["rho"].unique())
    _means = _grouped.mean()
    _stds = _grouped.std().fillna(0)

    # Plot individual trials as scatter
    for _, _row in inf_df.iterrows():
        _ax.scatter(
            _row["rho"],
            _row["test_accuracy"],
            color="steelblue",
            alpha=0.5,
            s=60,
            zorder=3,
            label="Individual trials" if _ == inf_df.index[0] else "",
        )

    # Plot mean ± std as line
    _ax.errorbar(
        _rho_vals,
        _means[_rho_vals],
        yerr=_stds[_rho_vals],
        fmt="o-",
        color="navy",
        linewidth=2.5,
        markersize=9,
        capsize=5,
        capthick=2,
        zorder=4,
        label="Mean ± Std",
    )

    _ax.set_xlabel("ρ (Topographic Loss Weight)")
    _ax.set_ylabel("Test Accuracy")
    _ax.set_title("Test Accuracy vs Topographic Loss Weight (ρ)")
    _ax.set_xticks(_rho_vals)
    _ax.set_xticklabels([str(r) for r in _rho_vals])
    _ax.grid(True, alpha=0.3, linestyle="--")
    _ax.legend()

    for _r in _rho_vals:
        _ax.annotate(
            f"{_means[_r]:.4f}",
            xy=(_r, _means[_r]),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            color="navy",
            fontweight="bold",
        )

    _fig.tight_layout()
    mo.md("## 3. Accuracy vs ρ")
    return


@app.cell
def query_profile_runs():
    """Query diagnostics runs for Moran's I."""
    _runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "tags.kind = 'diagnostics' and "
            "tags.diagnostic_metric = 'morans_i' and "
            "attributes.status = 'FINISHED'"
        ),
    )

    if _runs.empty:
        prof_df = pd.DataFrame()
        mo.stop(
            True,
            mo.md(
                "⚠️ **No diagnostics runs found.** Run `python scripts/03b_compute_diagnostics.py` first."
            ),
        )

    prof_df = _runs[
        ["run_id", "tags.parent_run_id", "tags.rho", "tags.trial", "tags.topology"]
    ].copy()
    prof_df.columns = ["prof_run_id", "model_run_id", "rho", "trial", "topology"]
    prof_df["rho"] = prof_df["rho"].astype(float)
    prof_df["trial"] = prof_df["trial"].astype(int)

    _mi_col = [c for c in _runs.columns if "morans_i" in c.lower()]
    if _mi_col:
        prof_df["morans_i"] = _runs[_mi_col[0]].values

    mo.md(f"""
    ## 4. Profiling Metrics (Moran's I)

    Found **{len(prof_df)} diagnostics runs** with Moran's I data.

    {prof_df[["rho", "trial", "topology", "morans_i"]].sort_values(["rho", "trial"]).to_markdown(index=False)}
    """)
    return (prof_df,)


@app.cell
def plot_morans_i(prof_df):
    """Plot: Moran's I vs rho."""
    if prof_df.empty or "morans_i" not in prof_df.columns:
        mo.stop(True, mo.md("*Skipping Moran's I plot — no profile data available.*"))

    _fig, _ax = plt.subplots(figsize=(8, 5))

    _rho_vals = sorted(prof_df["rho"].unique())
    _grouped = prof_df.groupby("rho")["morans_i"]
    _means = _grouped.mean()
    _stds = _grouped.std().fillna(0)

    for _, _row in prof_df.iterrows():
        _ax.scatter(
            _row["rho"],
            _row["morans_i"],
            color="coral",
            alpha=0.5,
            s=60,
            zorder=3,
            label="Individual trials" if _ == prof_df.index[0] else "",
        )

    _ax.errorbar(
        _rho_vals,
        _means[_rho_vals],
        yerr=_stds[_rho_vals],
        fmt="s-",
        color="darkred",
        linewidth=2.5,
        markersize=9,
        capsize=5,
        capthick=2,
        zorder=4,
        label="Mean ± Std",
    )

    _ax.set_xlabel("ρ (Topographic Loss Weight)")
    _ax.set_ylabel("Moran's I (Spatial Autocorrelation)")
    _ax.set_title(
        "Moran's I vs ρ — Effect of Topographic Loss on Representation Smoothness"
    )
    _ax.set_xticks(_rho_vals)
    _ax.set_xticklabels([str(r) for r in _rho_vals])
    _ax.grid(True, alpha=0.3, linestyle="--")
    _ax.legend()

    for _r in _rho_vals:
        _ax.annotate(
            f"{_means[_r]:.4f}",
            xy=(_r, _means[_r]),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            color="darkred",
            fontweight="bold",
        )

    _fig.tight_layout()
    mo.md("## 5. Spatial Smoothness (Moran's I) vs ρ")
    return


@app.cell
def load_embeddings(inf_df):
    """Load embeddings from local artifact cache for all models."""
    embeddings_by_run = {}
    for _, _row in inf_df.iterrows():
        _run_id = _row["model_run_id"]
        _emb_path = os.path.join(
            ARTIFACTS_ROOT, "inference", _run_id, "test", "embeddings.pt"
        )
        if os.path.isfile(_emb_path):
            _emb = torch.load(_emb_path, weights_only=True)
            embeddings_by_run[_run_id] = {
                "embeddings": _emb,
                "rho": _row["rho"],
                "trial": _row["trial"],
                "topology": _row["topology"],
            }

    mo.md(f"""
    ## 6. Embedding Analysis

    Loaded embeddings for **{len(embeddings_by_run)} models**.
    Embedding shape: `{list(embeddings_by_run.values())[0]['embeddings'].shape if embeddings_by_run else 'N/A'}`
    """)
    return (embeddings_by_run,)


@app.cell
def embedding_similarity_heatmaps(embeddings_by_run):
    """Cross-model embedding cosine similarity."""
    import torch.nn.functional as F

    if not embeddings_by_run:
        mo.stop(True, mo.md("*No embeddings loaded.*"))

    _run_ids = list(embeddings_by_run.keys())
    _n = len(_run_ids)

    _mean_embs = []
    _labels = []
    for _rid in _run_ids:
        _info = embeddings_by_run[_rid]
        _mean_embs.append(_info["embeddings"].float().mean(dim=0))
        _labels.append(f"ρ={_info['rho']}\ntrial={_info['trial']}")

    _mean_embs = torch.stack(_mean_embs)
    _mean_embs_norm = F.normalize(_mean_embs, p=2, dim=1)
    _cos_sim = (_mean_embs_norm @ _mean_embs_norm.T).numpy()

    _fig, _ax = plt.subplots(figsize=(6, 5))
    _im = _ax.imshow(_cos_sim, cmap="RdYlBu_r", vmin=_cos_sim.min() * 0.95, vmax=1.0)
    _ax.set_xticks(range(_n))
    _ax.set_yticks(range(_n))
    _ax.set_xticklabels(_labels, fontsize=9)
    _ax.set_yticklabels(_labels, fontsize=9)
    _ax.set_title("Cross-Model Mean Embedding Cosine Similarity")

    for _i in range(_n):
        for _j in range(_n):
            _ax.text(
                _j,
                _i,
                f"{_cos_sim[_i, _j]:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if _cos_sim[_i, _j] < 0.95 else "black",
            )

    _fig.colorbar(_im, ax=_ax, label="Cosine Similarity")
    _fig.tight_layout()
    mo.md("## 7. Cross-Model Embedding Similarity")
    return


@app.cell
def prediction_agreement(inf_df):
    """Per-sample prediction agreement across trials within same rho."""
    _agreement_stats = []
    for _rho in sorted(inf_df["rho"].unique()):
        _rho_runs = inf_df[inf_df["rho"] == _rho]
        if len(_rho_runs) < 2:
            continue

        _preds_list = []
        for _, _row in _rho_runs.iterrows():
            _preds_path = os.path.join(
                ARTIFACTS_ROOT, "inference", _row["model_run_id"], "test", "preds.pt"
            )
            if os.path.isfile(_preds_path):
                _preds_list.append(torch.load(_preds_path, weights_only=True))

        if len(_preds_list) < 2:
            continue

        _preds = torch.stack(_preds_list)
        _all_agree = (_preds == _preds[0:1]).all(dim=0).float().mean().item()
        _n_trials = _preds.size(0)
        _pair_agrees = []
        for _i in range(_n_trials):
            for _j in range(_i + 1, _n_trials):
                _pair_agrees.append((_preds[_i] == _preds[_j]).float().mean().item())
        _mean_pair = np.mean(_pair_agrees)

        _agreement_stats.append(
            {
                "rho": _rho,
                "unanimous_agreement": _all_agree,
                "mean_pairwise_agreement": _mean_pair,
                "num_trials": _n_trials,
            }
        )

    if not _agreement_stats:
        mo.stop(True, mo.md("*Need at least 2 trials per rho for agreement analysis.*"))

    _agree_df = pd.DataFrame(_agreement_stats)

    _fig, _ax = plt.subplots(figsize=(8, 5))
    _x = range(len(_agree_df))
    _width = 0.35

    _bars1 = _ax.bar(
        [xi - _width / 2 for xi in _x],
        _agree_df["unanimous_agreement"],
        _width,
        label="Unanimous Agreement",
        color="steelblue",
        alpha=0.8,
    )
    _bars2 = _ax.bar(
        [xi + _width / 2 for xi in _x],
        _agree_df["mean_pairwise_agreement"],
        _width,
        label="Mean Pairwise Agreement",
        color="coral",
        alpha=0.8,
    )

    _ax.set_xlabel("ρ (Topographic Loss Weight)")
    _ax.set_ylabel("Agreement Rate")
    _ax.set_title("Prediction Agreement Across Trials")
    _ax.set_xticks(_x)
    _ax.set_xticklabels([str(r) for r in _agree_df["rho"]])
    _ax.legend()
    _ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    _ax.set_ylim(0.8, 1.0)

    for _bar in _bars1:
        _ax.annotate(
            f"{_bar.get_height():.3f}",
            xy=(_bar.get_x() + _bar.get_width() / 2, _bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    for _bar in _bars2:
        _ax.annotate(
            f"{_bar.get_height():.3f}",
            xy=(_bar.get_x() + _bar.get_width() / 2, _bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    _fig.tight_layout()
    mo.md(f"""
    ## 8. Prediction Agreement Across Trials

    {_agree_df.to_markdown(index=False)}
    """)
    return


@app.cell
def load_logits_and_confidence(inf_df):
    """Confidence analysis: how does rho affect model confidence?"""
    _conf_stats = []
    for _, _row in inf_df.iterrows():
        _probs_path = os.path.join(
            ARTIFACTS_ROOT, "inference", _row["model_run_id"], "test", "probs.pt"
        )
        _labels_path = os.path.join(
            ARTIFACTS_ROOT, "inference", _row["model_run_id"], "test", "labels.pt"
        )
        if not os.path.isfile(_probs_path) or not os.path.isfile(_labels_path):
            continue

        _probs = torch.load(_probs_path, weights_only=True)
        _labels = torch.load(_labels_path, weights_only=True)
        _preds = _probs.argmax(dim=1)

        _max_conf = _probs.max(dim=1).values
        _correct = _preds == _labels

        _conf_stats.append(
            {
                "rho": _row["rho"],
                "trial": _row["trial"],
                "mean_confidence": _max_conf.mean().item(),
                "median_confidence": _max_conf.median().item(),
                "confidence_correct": _max_conf[_correct].mean().item(),
                "confidence_incorrect": (
                    _max_conf[~_correct].mean().item()
                    if (~_correct).sum() > 0
                    else float("nan")
                ),
                "entropy_mean": (-_probs * _probs.clamp_min(1e-8).log())
                .sum(dim=1)
                .mean()
                .item(),
            }
        )

    if not _conf_stats:
        mo.stop(True, mo.md("*No probability data available.*"))

    _conf_df = pd.DataFrame(_conf_stats)

    _fig, _axes = plt.subplots(1, 3, figsize=(16, 5))

    _grouped = _conf_df.groupby("rho")
    _rho_vals = sorted(_conf_df["rho"].unique())

    _ax = _axes[0]
    _means_correct = _grouped["confidence_correct"].mean()
    _means_incorrect = _grouped["confidence_incorrect"].mean()
    _ax.plot(
        _rho_vals,
        _means_correct[_rho_vals],
        "o-",
        color="forestgreen",
        linewidth=2.5,
        markersize=8,
        label="Correct predictions",
    )
    _ax.plot(
        _rho_vals,
        _means_incorrect[_rho_vals],
        "s-",
        color="crimson",
        linewidth=2.5,
        markersize=8,
        label="Incorrect predictions",
    )
    _ax.set_xlabel("ρ")
    _ax.set_ylabel("Mean Max Probability")
    _ax.set_title("Confidence by Correctness")
    _ax.legend()
    _ax.grid(True, alpha=0.3, linestyle="--")
    _ax.set_xticks(_rho_vals)

    _ax = _axes[1]
    _means_entropy = _grouped["entropy_mean"].mean()
    _stds_entropy = _grouped["entropy_mean"].std().fillna(0)
    _ax.errorbar(
        _rho_vals,
        _means_entropy[_rho_vals],
        yerr=_stds_entropy[_rho_vals],
        fmt="D-",
        color="purple",
        linewidth=2.5,
        markersize=8,
        capsize=5,
        capthick=2,
    )
    _ax.set_xlabel("ρ")
    _ax.set_ylabel("Mean Prediction Entropy (nats)")
    _ax.set_title("Prediction Uncertainty")
    _ax.grid(True, alpha=0.3, linestyle="--")
    _ax.set_xticks(_rho_vals)

    _ax = _axes[2]
    _ax.axis("off")
    _table_data = []
    for _rho in _rho_vals:
        _rho_data = _conf_df[_conf_df["rho"] == _rho]
        _table_data.append(
            [
                f"{_rho}",
                f"{_rho_data['mean_confidence'].mean():.4f}",
                f"{_rho_data['confidence_correct'].mean():.4f}",
                f"{_rho_data['confidence_incorrect'].mean():.4f}",
                f"{_rho_data['entropy_mean'].mean():.4f}",
            ]
        )
    _table = _ax.table(
        cellText=_table_data,
        colLabels=["ρ", "Mean Conf", "Conf (✓)", "Conf (✗)", "Entropy"],
        loc="center",
        cellLoc="center",
    )
    _table.auto_set_font_size(False)
    _table.set_fontsize(10)
    _table.scale(1, 1.5)
    _ax.set_title("Summary", fontsize=12, pad=20)

    _fig.suptitle("Prediction Confidence & Uncertainty Analysis", fontsize=14, y=1.02)
    _fig.tight_layout()
    mo.md("## 9. Confidence & Uncertainty Analysis")
    return


@app.cell
def summary(inf_df, prof_df):
    """Final summary table combining all metrics."""
    _summary_data = []

    for _rho in sorted(inf_df["rho"].unique()):
        _rho_inf = inf_df[inf_df["rho"] == _rho]
        _entry = {
            "ρ": _rho,
            "Num Trials": len(_rho_inf),
            "Mean Accuracy": f"{_rho_inf['test_accuracy'].mean():.4f}",
            "Std Accuracy": (
                f"{_rho_inf['test_accuracy'].std():.4f}" if len(_rho_inf) > 1 else "—"
            ),
        }
        if not prof_df.empty and "morans_i" in prof_df.columns:
            _rho_prof = prof_df[prof_df["rho"] == _rho]
            if not _rho_prof.empty:
                _entry["Mean Moran's I"] = f"{_rho_prof['morans_i'].mean():.4f}"
        _summary_data.append(_entry)

    _summary_df = pd.DataFrame(_summary_data)

    mo.md(f"""
    ## 10. Summary

    {_summary_df.to_markdown(index=False)}

    ---

    *Generated from MLflow experiment data. Re-run cells after new pipeline steps to refresh.*
    """)
    return


if __name__ == "__main__":
    app.run()
