# %%
import polars as pl
import plotly.express as px

from src.config.notebook import setup_environment

cfg, experiment = setup_environment()

import notebooks.mlflow.mlflow_helpers as mh
print("experiment:", experiment.name)

# %%
training_runs = mh.get_base_model_list(experiment).select(["run_id", "tags.rho"])
print(f"Found {training_runs.height} training runs with 'rho' tag.")

inference_runs = mh.get_inference_list(experiment).select(
    ["tags.trained_model_run_id", "metrics.accuracy"]
)

# %%
merged_runs = (
    training_runs.join(
        inference_runs,
        left_on="run_id",
        right_on="tags.trained_model_run_id",
        how="inner",
    )
    .group_by("tags.rho")
    .agg(pl.col("metrics.accuracy").mean().alias("avg_accuracy"))
    .rename({"tags.rho": "rho"})
    .sort("rho")
)

print(merged_runs.head())

# %%
fig = px.scatter(
    merged_runs,
    x="rho",
    y="avg_accuracy",
    title="Average Inference Accuracy by Rho",
    labels={"rho": "Rho", "avg_accuracy": "Average Accuracy"},
)
fig.update_traces(marker=dict(size=10, opacity=0.8))
fig.update_layout(template="simple_white")
fig.show()
