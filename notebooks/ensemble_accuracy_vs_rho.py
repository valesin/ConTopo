# %% [markdown]
# # Ensemble Accuracy vs Rho
# This notebook visualizes the relationship between the parameter rho and two accuracy metrics
# for ensembles: the mean component accuracy (union) and the mean ensemble accuracy,
# both averaged over all sampled combinations per rho.

# %%
import polars as pl
import plotly.graph_objects as go

from src.config.notebook import setup_environment

cfg, experiment = setup_environment()

import notebooks.mlflow_helpers as mh

print("experiment:", experiment.name)

# %%
NUM_COMPONENTS = 9  # number of components per sampled ensemble
VOTE_METHOD = "soft"

ensembles = mh.get_ensemble_list(experiment)
ensembles = ensembles.filter(pl.col("params.method") == VOTE_METHOD)
ensembles = ensembles.filter(pl.col("params.num_components") == str(NUM_COMPONENTS))

# Aggregate over all combinations per rho.
# mean_comp_union_acc: because each model appears in the same number of combinations,
# averaging comp_mean_acc across all combos equals the mean accuracy of the full
# component union for that rho.
# mean_gain: average of per-combo (ensemble_acc - comp_mean_acc).
plot_df = (
    ensembles.with_columns(
        pl.col("tags.rho").cast(pl.Float64).alias("rho_numeric"),
        (pl.col("metrics.ensemble_accuracy") - pl.col("metrics.comp_mean_acc")).alias(
            "gain"
        ),
    )
    .group_by("tags.rho", "rho_numeric")
    .agg(
        [
            pl.col("metrics.ensemble_accuracy").mean().alias("mean_ensemble_acc"),
            pl.col("metrics.comp_mean_acc").mean().alias("mean_comp_union_acc"),
            pl.col("gain").mean().alias("mean_gain"),
            pl.len().alias("n_combinations"),
        ]
    )
    .sort("rho_numeric")
)

# %%
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=plot_df["tags.rho"],
        y=plot_df["mean_comp_union_acc"],
        mode="lines+markers",
        name="Component Accuracy — Union (mean)",
    )
)
fig.add_trace(
    go.Scatter(
        x=plot_df["tags.rho"],
        y=plot_df["mean_ensemble_acc"],
        mode="lines+markers",
        name="Ensemble Accuracy (mean over combinations)",
    )
)
fig.update_layout(
    title=f"Ensemble and Component Accuracy vs Rho (k={NUM_COMPONENTS}, vote={VOTE_METHOD})",
    xaxis_title="Rho",
    yaxis_title="Accuracy",
    template="simple_white",
)
fig.show()

# %%
gain_df = plot_df.select(
    [
        "tags.rho",
        "mean_comp_union_acc",
        "mean_ensemble_acc",
        "mean_gain",
    ]
).rename(
    {
        "tags.rho": "ρ",
        "mean_comp_union_acc": "component mean (union)",
        "mean_ensemble_acc": "ensemble acc",
        "mean_gain": "gain (ens − comp, avg)",
    }
)

fig_table = go.Figure(
    go.Table(
        header=dict(
            values=list(gain_df.columns),
            align="left",
            font=dict(size=13),
        ),
        cells=dict(
            values=[gain_df[c].to_list() for c in gain_df.columns],
            align="left",
            format=["", ".4f", ".4f", "+.4f"],
        ),
    )
)
fig_table.update_layout(
    title=f"Performance gain from ensembling per ρ (k={NUM_COMPONENTS}, averaged over combinations)",
    margin=dict(t=50, b=10, l=10, r=10),
    height=60 + 30 * gain_df.height,
)
fig_table.show()

# Save the gain table as a LaTeX file next to saved_plots in notebooks/mlflow/saved_tables
import os
import pandas as _pd

tex_df = gain_df.to_pandas()
tex_str = tex_df.to_latex(index=False, float_format="%.4f", escape=False)

out_dir = os.path.join(os.path.dirname(__file__), "saved_tables")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "ensemble_gain_table.tex")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(tex_str)

print(f"Saved LaTeX table to {out_path}")
