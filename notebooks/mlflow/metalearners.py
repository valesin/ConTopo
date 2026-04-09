# %% [markdown]
# # Metalearners Analysis

# %% CONFIG
VOTE_METHOD = "soft"
FEATURE_TYPE = "embeddings+profiles"
SIM_METRIC = "cosine"

# %%
import polars as pl
import plotly.express as px

from src.config.notebook import setup_environment

cfg, experiment = setup_environment()

from notebooks.mlflow.mlflow_helpers import (
    get_metalearner_results,
    get_ensemble_results,
)

print("experiment:", experiment.name)

# %%
# Normalized columns:
#   metalearners: run_id, rho, rho_numeric, cs_hash, meta_type, feature_type,
#                 similarity_metric, split_seed, ensemble_name, profile_mask, accuracy
#   ensembles:    run_id, rho, rho_numeric, cs_hash, vote_method, ensemble_name,
#                 accuracy, comp_mean_acc
ml_df = get_metalearner_results(experiment.name)
ens_df = get_ensemble_results(experiment.name, split="test")

# %%
ml_filtered = ml_df[
    (ml_df["feature_type"] == FEATURE_TYPE) & (ml_df["similarity_metric"] == SIM_METRIC)
]
ens_filtered = ens_df[ens_df["vote_method"] == VOTE_METHOD]

print("metalearners:", len(ml_filtered), "| ensembles:", len(ens_filtered))

# %%
# Attach ensemble_accuracy to each metalearner row via cs_hash,
# then melt so both metrics share a single "accuracy" column for plotting.
ens_acc = ens_filtered[["cs_hash", "rho_numeric", "accuracy"]].rename(
    {"accuracy": "ensemble_accuracy"}
)

merged = ml_filtered.merge(ens_acc, on=["cs_hash", "rho_numeric"], how="left")

plot_df = merged.melt(
    id_vars=["rho_numeric", "meta_type"],
    value_vars=["accuracy", "ensemble_accuracy"],
    var_name="metric_type",
    value_name="accuracy_val",
)

plot_df["line_type"] = plot_df.apply(
    lambda r: (
        "Ensemble Accuracy"
        if r["metric_type"] == "ensemble_accuracy"
        else r["meta_type"]
    ),
    axis=1,
)

plot_df = plot_df.sort_values("rho_numeric")

# %%
fig = px.line(
    plot_df,
    x="rho_numeric",
    y="accuracy_val",
    color="line_type",
    markers=True,
    title="Meta Inference and Ensemble Accuracy by Rho",
    labels={"rho_numeric": "Rho", "accuracy_val": "Accuracy", "line_type": "Type"},
)
fig.update_traces(marker=dict(size=10, opacity=0.8))
fig.update_layout(template="simple_white")
fig.show()
