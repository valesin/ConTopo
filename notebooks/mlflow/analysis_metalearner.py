# %% [markdown]
# # Meta-learner accuracy vs ρ
#
# Compares holdout accuracy of each meta-learner type and feature combination
# against the soft-ensemble baseline and component mean accuracy across
# all values of the topographic regularisation weight ρ.

# %% CONFIG
SPLIT = "test"          # inference split used by ensemble runs
VOTE_METHOD = "soft"    # ensemble voting method to use as baseline
SIM_METRIC = "cosine"   # similarity metric filter for metalearner runs
PROFILE_MASK = "all"    # profile_mask filter for metalearner runs (or None to skip)

# %% SETUP
from src.config.notebook import setup_environment
from notebooks.mlflow.mlflow_helpers import get_metalearner_results, get_ensemble_results

cfg, experiment = setup_environment()
print(f"Experiment: {experiment.name}")

# %% LOAD
ml_df = get_metalearner_results(experiment.name)
ens_df = get_ensemble_results(experiment.name, split=SPLIT)

# %% INSPECT
print(f"Metalearner runs:  {len(ml_df)}")
print(f"Ensemble runs:     {len(ens_df)}")

if not ml_df.empty:
    print("\nUnique rho values:     ", sorted(ml_df["rho_numeric"].dropna().unique().tolist()))
    print("Unique feature_type:   ", sorted(ml_df["feature_type"].dropna().unique().tolist()))
    print("Unique meta_type:      ", sorted(ml_df["meta_type"].dropna().unique().tolist()))
    if "profile_mask" in ml_df.columns:
        print("Unique profile_mask:   ", sorted(ml_df["profile_mask"].dropna().unique().tolist()))

if not ens_df.empty:
    print("\nUnique vote_method:    ", sorted(ens_df["vote_method"].dropna().unique().tolist()))

# %% FILTER
import pandas as pd

ml_filtered = ml_df.copy()
if SIM_METRIC is not None and "similarity_metric" in ml_filtered.columns:
    ml_filtered = ml_filtered[ml_filtered["similarity_metric"] == SIM_METRIC]
if PROFILE_MASK is not None and "profile_mask" in ml_filtered.columns:
    ml_filtered = ml_filtered[ml_filtered["profile_mask"] == PROFILE_MASK]

ens_filtered = ens_df[ens_df["vote_method"] == VOTE_METHOD] if not ens_df.empty else ens_df

print(f"Metalearner rows after filter: {len(ml_filtered)}")
print(f"Ensemble rows after filter:    {len(ens_filtered)}")

# %% AGGREGATE
# Metalearner: mean ± std of holdout accuracy per (rho_numeric, feature_type, meta_type)
if not ml_filtered.empty:
    ml_agg = (
        ml_filtered
        .groupby(["rho_numeric", "feature_type", "meta_type"])["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "acc_mean", "std": "acc_std"})
    )
    ml_agg["acc_std"] = ml_agg["acc_std"].fillna(0)
else:
    ml_agg = pd.DataFrame(columns=["rho_numeric", "feature_type", "meta_type", "acc_mean", "acc_std"])

# Ensemble baseline: mean ± std of accuracy per rho_numeric
if not ens_filtered.empty:
    ens_agg = (
        ens_filtered
        .groupby("rho_numeric")["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "ens_mean", "std": "ens_std"})
    )
    ens_agg["ens_std"] = ens_agg["ens_std"].fillna(0)

    # Component mean accuracy baseline
    if "comp_mean_acc" in ens_filtered.columns:
        comp_agg = (
            ens_filtered
            .groupby("rho_numeric")["comp_mean_acc"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "comp_mean", "std": "comp_std"})
        )
        comp_agg["comp_std"] = comp_agg["comp_std"].fillna(0)
    else:
        comp_agg = None
else:
    ens_agg = pd.DataFrame()
    comp_agg = None

print(f"\nMetalearner aggregated rows: {len(ml_agg)}")
print(ml_agg.to_string(index=False) if not ml_agg.empty else "  (empty)")

# %% PLOT
import plotly.graph_objects as go

RHO_TICKS = [0, 0.008, 0.04, 0.2, 1.0, 5.0]
RHO_TICK_TEXT = ["0", "0.008", "0.04", "0.2", "1.0", "5.0"]

fig = go.Figure()

# One trace per (feature_type, meta_type)
if not ml_agg.empty:
    groups = ml_agg.groupby(["feature_type", "meta_type"])
    for (feat, mtype), grp in groups:
        grp = grp.sort_values("rho_numeric")
        label = f"{feat} / {mtype}"
        x = grp["rho_numeric"].tolist()
        y = grp["acc_mean"].tolist()
        err = grp["acc_std"].tolist()

        fig.add_trace(go.Scatter(
            x=x, y=y, name=label, mode="lines+markers",
            error_y=dict(type="data", array=err, visible=True),
        ))

# Soft ensemble baseline
if not ens_agg.empty:
    ens_agg = ens_agg.sort_values("rho_numeric")
    fig.add_trace(go.Scatter(
        x=ens_agg["rho_numeric"].tolist(),
        y=ens_agg["ens_mean"].tolist(),
        name=f"ensemble ({VOTE_METHOD})",
        mode="lines+markers",
        line=dict(color="black", dash="dash"),
        error_y=dict(
            type="data", array=ens_agg["ens_std"].tolist(), visible=True,
            color="black",
        ),
    ))

# Component mean baseline
if comp_agg is not None:
    comp_agg = comp_agg.sort_values("rho_numeric")
    fig.add_trace(go.Scatter(
        x=comp_agg["rho_numeric"].tolist(),
        y=comp_agg["comp_mean"].tolist(),
        name="component mean",
        mode="lines+markers",
        line=dict(color="gray", dash="dot"),
        error_y=dict(
            type="data", array=comp_agg["comp_std"].tolist(), visible=True,
            color="gray",
        ),
    ))

fig.update_layout(
    title="Meta-learner holdout accuracy vs ρ",
    xaxis=dict(
        title="ρ (topographic regularisation weight)",
        type="log",
        tickvals=RHO_TICKS,
        ticktext=RHO_TICK_TEXT,
    ),
    yaxis=dict(title="Accuracy"),
    legend=dict(orientation="v", x=1.02, xanchor="left"),
    template="plotly_white",
)

# %% SHOW
fig.show()
