# %% [markdown]
# # Meta-learner accuracy vs ρ
#
# Compares holdout accuracy of each meta-learner type and feature combination
# against the soft-ensemble baseline and component mean accuracy across
# all values of the topographic regularisation weight ρ.

# %% CONFIG
SPLIT = "test"          # inference split used by ensemble runs
VOTE_METHOD = "soft"    # ensemble voting method to use as baseline
SIM_METRIC = "cosine"   # similarity metric filter for metalearner runs (None = no filter)

# %% SETUP
from src.config.notebook import setup_environment

cfg, experiment = setup_environment()

from notebooks.mlflow.mlflow_helpers import get_metalearner_results, get_ensemble_results, save_plot
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

# Drop runs that predate the profile_mask parameter
if "profile_mask" in ml_filtered.columns:
    ml_filtered = ml_filtered[ml_filtered["profile_mask"] != "N/A"]

if SIM_METRIC is not None and "similarity_metric" in ml_filtered.columns:
    ml_filtered = ml_filtered[ml_filtered["similarity_metric"] == SIM_METRIC]

ens_filtered = ens_df[ens_df["vote_method"] == VOTE_METHOD] if not ens_df.empty else ens_df

print(f"Metalearner rows after filter: {len(ml_filtered)}")
print(f"Ensemble rows after filter:    {len(ens_filtered)}")

if not ml_filtered.empty:
    group_cols = ["meta_type", "feature_type", "profile_mask"] if "profile_mask" in ml_filtered.columns else ["meta_type", "feature_type"]
    counts = (
        ml_filtered.groupby(group_cols)
        .size()
        .reset_index(name="n")
        .sort_values(group_cols)
    )
    for _, row in counts.iterrows():
        label = " / ".join(str(row[c]) for c in group_cols)
        print(f"  {label}, n={row['n']}")

# %% AGGREGATE
# Metalearner: mean ± std per (rho_numeric, meta_type, feature_type, profile_mask)
GROUP_COLS = ["rho_numeric", "meta_type", "feature_type"]
if not ml_filtered.empty and "profile_mask" in ml_filtered.columns:
    GROUP_COLS.append("profile_mask")

if not ml_filtered.empty:
    ml_agg = (
        ml_filtered
        .groupby(GROUP_COLS)["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "acc_mean", "std": "acc_std"})
    )
    ml_agg["acc_std"] = ml_agg["acc_std"].fillna(0)
else:
    ml_agg = pd.DataFrame(columns=GROUP_COLS + ["acc_mean", "acc_std"])

# Ensemble baseline: mean ± std per rho_numeric
if not ens_filtered.empty:
    ens_agg = (
        ens_filtered
        .groupby("rho_numeric")["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "ens_mean", "std": "ens_std"})
    )
    ens_agg["ens_std"] = ens_agg["ens_std"].fillna(0)

    comp_agg = None
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
    ens_agg = pd.DataFrame()
    comp_agg = None

print(f"\nMetalearner aggregated rows: {len(ml_agg)}")
print(ml_agg.to_string(index=False) if not ml_agg.empty else "  (empty)")

# %% PLOT
import plotly.graph_objects as go
import plotly.express as _px

RHO_TICKS = [0, 0.008, 0.04, 0.2, 1.0, 5.0]
RHO_TICK_TEXT = ["0", "0.008", "0.04", "0.2", "1.0", "5.0"]

fig = go.Figure()

# Jitter: on a log-scale x-axis, apply a small multiplicative offset per trace
# so error bars at the same nominal rho don't overlap.
label_cols = [c for c in ["meta_type", "feature_type", "profile_mask"] if c in ml_agg.columns]
_colors = _px.colors.qualitative.Plotly

if not ml_agg.empty:
    n_groups = ml_agg.groupby(label_cols).ngroups
    # Spread traces symmetrically around the nominal x; total span ~±8% in log space
    jitter_factors = [1.0 + (i - (n_groups - 1) / 2) * 0.04 for i in range(n_groups)]

    for i, (key, grp) in enumerate(ml_agg.groupby(label_cols)):
        key = (key,) if isinstance(key, str) else key
        label = " / ".join(str(k) for k in key)
        grp = grp.sort_values("rho_numeric")
        x = [v * jitter_factors[i] if v > 0 else v for v in grp["rho_numeric"].tolist()]
        y_mean = grp["acc_mean"].tolist()
        y_std = grp["acc_std"].tolist()
        color = _colors[i % len(_colors)]

        fig.add_trace(go.Scatter(
            x=x, y=y_mean, name=label, mode="lines+markers",
            line=dict(color=color), marker=dict(color=color, size=7),
            error_y=dict(type="data", array=y_std, visible=True, color=color, thickness=1.5),
        ))

# Soft ensemble baseline with error bars (no jitter — baseline sits at nominal x)
if not ens_agg.empty:
    ens_agg = ens_agg.sort_values("rho_numeric")
    fig.add_trace(go.Scatter(
        x=ens_agg["rho_numeric"].tolist(),
        y=ens_agg["ens_mean"].tolist(),
        name=f"ensemble ({VOTE_METHOD})",
        mode="lines+markers",
        line=dict(color="black", dash="dash"),
        marker=dict(color="black", size=7),
        error_y=dict(
            type="data", array=ens_agg["ens_std"].tolist(),
            visible=True, color="black", thickness=1.5,
        ),
    ))

# Component mean baseline with error bars
if comp_agg is not None:
    comp_agg = comp_agg.sort_values("rho_numeric")
    fig.add_trace(go.Scatter(
        x=comp_agg["rho_numeric"].tolist(),
        y=comp_agg["comp_mean"].tolist(),
        name="component mean",
        mode="lines+markers",
        line=dict(color="gray", dash="dot"),
        marker=dict(color="gray", size=7),
        error_y=dict(
            type="data", array=comp_agg["comp_std"].tolist(),
            visible=True, color="gray", thickness=1.5,
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
fig.show()
save_plot(fig, "analysis_metalearner")
