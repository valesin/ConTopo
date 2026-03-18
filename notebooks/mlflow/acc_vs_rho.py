# type: ignore
# %%
import mlflow
import polars as pl

# Set the backend store to your local SSH tunnel endpoint
mlflow.set_tracking_uri("http://localhost:5000")
# %%
# List all experiments to verify connection
# experiments = mlflow.search_experiments(view_type=mlflow.entities.ViewType.ALL)
experiment = mlflow.get_experiment_by_name("contopo")
models_pd = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id], filter_string="tags.kind = 'model'"
)
# Convert to Polars and select columns
training_runs = pl.from_pandas(models_pd).select(["run_id", "tags.rho"])
print(f"Found {training_runs.height} training runs with 'rho' tag.")
# %%
inference_runs_pd = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id], filter_string="tags.kind = 'inference'"
)
inference_runs = pl.from_pandas(inference_runs_pd).select(
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
    .rename({"tags.rho": "rho"})  # Clean up the column name
    .sort("rho")
)

print(merged_runs.head())
# %%
import plotly.express as px

# Create an interactive scatter plot
fig = px.scatter(
    merged_runs,
    x="rho",
    y="avg_accuracy",
    title="Average Inference Accuracy by Rho",
    labels={"rho": "Rho", "avg_accuracy": "Average Accuracy"},
)

# Optional: Improve the visual styling
fig.update_traces(marker=dict(size=10, opacity=0.8))
fig.update_layout(template="simple_white")

fig.show()
