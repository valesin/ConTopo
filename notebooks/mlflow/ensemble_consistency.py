# %%
import polars as pl

from src.config.notebook import setup_environment

cfg, experiment = setup_environment()

import notebooks.mlflow.mlflow_helpers as mh
print("experiment:", experiment.name)

# %%
ensembles = mh.get_ensemble_list(experiment)
ensembles = ensembles.filter(pl.col("params.method") == "soft")
ensembles = ensembles.filter(pl.col("params.num_components") == "10")
ensembles

# %%
ensembles.columns

# %%
cons = mh.get_consistency_list(experiment)
print("consistency runs:", cons.height)
cons.columns

# %%
if cons.height == 0 or "tags.component_set_hash" not in cons.columns:
    print("No consistency runs found — skipping join.")
else:
    merged = ensembles.join(cons, on="tags.component_set_hash", how="inner", suffix="_cons")
    print("merged rows:", merged.height)
    display(merged.columns)

    ens_cons = merged[["run_id", "run_id_cons", "tags.rho", "metrics.mean_rsa_correlation"]]
    display(ens_cons.sort("tags.rho"))
