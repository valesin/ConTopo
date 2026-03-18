# type: ignore
# %%
import mlflow
import polars as pl
import json

# Set the backend store to your local SSH tunnel endpoint
mlflow.set_tracking_uri("http://localhost:5000")
# %%
# Retrieve ensembles
experiment = mlflow.get_experiment_by_name("contopo")
models_pd = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id], filter_string="tags.kind = 'ensemble'"
)
ensembles = pl.from_pandas(models_pd)
ensembles = ensembles.filter(pl.col("params.method") == "soft")
ensembles
# %%
ensembles.columns
# %%

# %%
cons_pd = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id], filter_string="tags.kind = 'consistency'"
)
cons = pl.from_pandas(cons_pd)
cons.columns
# %%
merged = ensembles.join(cons, on="tags.component_set_hash", how="inner", suffix="_cons")
merged.columns
# %%
ens_cons = merged[["run_id", "run_id_cons", "tags.rho", "metrics.mean_rsa_correlation"]]
ens_cons.sort("tags.rho")
# %%
