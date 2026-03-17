# type: ignore
# %%
import mlflow
import polars as pl

# Set the backend store to your local SSH tunnel endpoint
mlflow.set_tracking_uri("http://localhost:5000")
# %%
# Retrieve ensembles
experiment = mlflow.get_experiment_by_name("contopo")
models_pd = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id], filter_string="tags.kind = 'ensemble'"
)
ensembles = pl.from_pandas(models_pd)
# %%
# run_id
# %%
run_id = ensembles["run_id"][0]  # Replace with your ensemble run ID
artifact_path = "ensemble_data/composition_map.json"

local_path = mlflow.artifacts.download_artifacts(
    run_id=run_id, artifact_path=artifact_path
)
with open(local_path, "r") as f:
    composition_map = json.load(f)  # %%
