# %% [markdown]
# # Ensemble Accuracy vs Rho
# This notebook visualizes the relationship between the parameter rho and two accuracy metrics
# for ensembles: the mean component accuracy and the ensemble accuracy.

# %%
import polars as pl
import plotly.graph_objects as go

from src.config.notebook import setup_environment

cfg, experiment = setup_environment()

import notebooks.mlflow.mlflow_helpers as mh

print("experiment:", experiment.name)

# %%
ensembles = mh.get_ensemble_list(experiment)
ensembles = ensembles.filter(pl.col("params.method") == "soft")
ensembles = ensembles.filter(pl.col("params.num_components") == "10")

# Select relevant columns
plot_df = ensembles.select(
    ["tags.rho", "metrics.comp_mean_acc", "metrics.ensemble_accuracy"]
)
plot_df = plot_df.sort("tags.rho")

# %%
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=plot_df["tags.rho"],
        y=plot_df["metrics.comp_mean_acc"],
        mode="lines+markers",
        name="Mean Component Accuracy",
    )
)
fig.add_trace(
    go.Scatter(
        x=plot_df["tags.rho"],
        y=plot_df["metrics.ensemble_accuracy"],
        mode="lines+markers",
        name="Ensemble Accuracy",
    )
)
fig.update_layout(
    title="Ensemble and Component Accuracy vs Rho",
    xaxis_title="Rho",
    yaxis_title="Accuracy",
    template="simple_white",
)
fig.show()

# %%
gain_df = (
    plot_df.with_columns(
        [
            pl.col("tags.rho").cast(pl.Float64).alias("rho_numeric"),
            (
                pl.col("metrics.ensemble_accuracy") - pl.col("metrics.comp_mean_acc")
            ).alias("gain"),
        ]
    )
    .sort("rho_numeric")
    .select(["tags.rho", "metrics.comp_mean_acc", "metrics.ensemble_accuracy", "gain"])
    .rename(
        {
            "tags.rho": "ρ",
            "metrics.comp_mean_acc": "component mean",
            "metrics.ensemble_accuracy": "ensemble acc",
            "gain": "gain (ens − comp)",
        }
    )
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
    title="Performance gain from ensembling per ρ",
    margin=dict(t=50, b=10, l=10, r=10),
    height=60 + 30 * gain_df.height,
)
fig_table.show()

# Save the gain table as a LaTeX file next to saved_plots in notebooks/mlflow/saved_tables
import os
import pandas as _pd

tex_df = gain_df.to_pandas()
# Use pandas to_latex for a nicely formatted LaTeX table
tex_str = tex_df.to_latex(index=False, float_format="%.4f", escape=False)

out_dir = os.path.join(os.path.dirname(__file__), "saved_tables")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "ensemble_gain_table.tex")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(tex_str)

print(f"Saved LaTeX table to {out_path}")
