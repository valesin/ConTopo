# %%
import polars as pl
import plotly.graph_objects as go

from src.config.notebook import setup_environment

cfg, experiment = setup_environment()

import notebooks.mlflow_helpers as mh

print("experiment:", experiment.name)

# %%
training_runs = mh.get_base_model_list(experiment).select(["run_id", "tags.rho"])
print(f"Found {training_runs.height} training runs with 'rho' tag.")

inference_runs = mh.get_inference_list(experiment).select(
    ["tags.trained_model_run_id", "metrics.accuracy"]
)

# %%
# Per-run accuracy joined with rho
per_run = (
    training_runs.join(
        inference_runs,
        left_on="run_id",
        right_on="tags.trained_model_run_id",
        how="inner",
    )
    .rename({"tags.rho": "rho", "metrics.accuracy": "accuracy"})
    .sort("rho")
)

# Aggregate: mean ± std per rho
agg = (
    per_run.group_by("rho")
    .agg(
        pl.col("accuracy").mean().alias("mean"),
        pl.col("accuracy").std().alias("std"),
        pl.col("accuracy").count().alias("n"),
    )
    .with_columns(pl.col("std").fill_null(0))
    .sort("rho")
)

print(agg)

# %%
fig = go.Figure()

# Strip: individual run points (jittered slightly via opacity)
fig.add_trace(
    go.Scatter(
        x=per_run["rho"].to_list(),
        y=per_run["accuracy"].to_list(),
        mode="markers",
        marker=dict(size=6, opacity=0.4, color="steelblue"),
        name="individual runs",
    )
)

# Mean line with error bars (± std)
fig.add_trace(
    go.Scatter(
        x=agg["rho"].to_list(),
        y=agg["mean"].to_list(),
        mode="lines+markers",
        marker=dict(size=10, color="darkblue"),
        line=dict(color="darkblue"),
        error_y=dict(
            type="data", array=agg["std"].to_list(), visible=True, color="darkblue"
        ),
        name="mean ± std",
    )
)

fig.update_layout(
    title="Inference Accuracy by Rho",
    xaxis_title="Rho",
    yaxis_title="Accuracy",
    template="simple_white",
)
fig.show()
mh.save_plot(fig, "acc_vs_rho")
