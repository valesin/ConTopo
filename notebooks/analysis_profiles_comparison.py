# %% [markdown]
# # Profile vs baseline metalearner comparison
#
# Compares metalearners trained with similarity-profile RDM features (non-leaky conditions)
# against baseline metalearners without profiles (e.g. stacked logits, stacked embeddings).
# Leaky profile masks (true_class) are excluded.

# %% CONFIG
SPLIT = "test"
VOTE_METHOD = "soft"
SIM_METRIC = "cosine"
META_TYPE = "meta_mlp_2"
FEATURE_TYPES = ["embeddings+profiles", "logits", "embeddings"]

# %% SETUP
import pandas as pd
import plotly.graph_objects as go
import plotly.express as _px

from src.config.notebook import setup_environment

cfg, experiment = setup_environment()

from notebooks.mlflow_helpers import (
    get_metalearner_results,
    get_ensemble_results,
    save_plot,
)

print(f"Experiment: {experiment.name}")

# %% LOAD
ml_df = get_metalearner_results(experiment.name)
ens_df = get_ensemble_results(experiment.name, split=SPLIT)
print(f"Metalearner runs: {len(ml_df)}")
print(f"Ensemble runs:    {len(ens_df)}")

# %% INSPECT
if not ml_df.empty:
    print(
        "feature_type:      ", sorted(ml_df["feature_type"].dropna().unique().tolist())
    )
    print("meta_type:         ", sorted(ml_df["meta_type"].dropna().unique().tolist()))
    if "profile_mask" in ml_df.columns:
        print(
            "profile_mask:      ",
            sorted(ml_df["profile_mask"].dropna().unique().tolist()),
        )
    if "similarity_metric" in ml_df.columns:
        print(
            "similarity_metric: ",
            sorted(ml_df["similarity_metric"].dropna().unique().tolist()),
        )

# %% FILTER
ml_filtered = ml_df.copy()

# Keep only the requested meta_type and feature_types
ml_filtered = ml_filtered[ml_filtered["meta_type"] == META_TYPE]
ml_filtered = ml_filtered[ml_filtered["feature_type"].isin(FEATURE_TYPES)]

# For embeddings+profiles only: drop legacy (N/A) and leaky (true_class) masks,
# and filter by similarity metric. logits/embeddings rows are never touched.
is_profile = ml_filtered["feature_type"] == "embeddings+profiles"

if "profile_mask" in ml_filtered.columns:
    ml_filtered = ml_filtered[
        ~is_profile
        | ml_filtered["profile_mask"].isin(
            [
                m
                for m in ml_filtered["profile_mask"].dropna().unique()
                if m not in ("N/A")
            ]
        )
    ]
    is_profile = (
        ml_filtered["feature_type"] == "embeddings+profiles"
    )  # recompute after filter

if SIM_METRIC is not None and "similarity_metric" in ml_filtered.columns:
    ml_filtered = ml_filtered[
        ~is_profile | (ml_filtered["similarity_metric"] == SIM_METRIC)
    ]

print(f"\nRows after filter: {len(ml_filtered)}")
if not ml_filtered.empty:
    group_cols = (
        ["feature_type", "profile_mask"]
        if "profile_mask" in ml_filtered.columns
        else ["feature_type"]
    )
    print(ml_filtered.groupby(group_cols, dropna=False).size().to_string())


# %% AGGREGATE
# Assign a line_label to each row: flattens feature_type + profile_mask into a
# single string so downstream aggregation and plotting need no NaN handling.
def _line_label(row: pd.Series) -> str:
    if row["feature_type"] == "embeddings+profiles":
        pm = row.get("profile_mask")
        return f"profiles / {pm}" if pd.notna(pm) else "profiles"
    return row["feature_type"]


ml_filtered = ml_filtered.copy()
ml_filtered["line_label"] = ml_filtered.apply(_line_label, axis=1)

ml_agg = (
    ml_filtered.groupby(["rho_numeric", "line_label"])["accuracy"]
    .agg(acc_mean="mean", acc_std="std")
    .reset_index()
)
ml_agg["acc_std"] = ml_agg["acc_std"].fillna(0)

print(f"\nAggregated rows: {len(ml_agg)}")
print(ml_agg.to_string(index=False) if not ml_agg.empty else "  (empty)")

# %% AGGREGATE BASELINES
ens_filtered = (
    ens_df[ens_df["vote_method"] == VOTE_METHOD] if not ens_df.empty else ens_df
)

ens_agg = pd.DataFrame()
comp_agg = None
if not ens_filtered.empty:
    ens_agg = (
        ens_filtered.groupby("rho_numeric")["accuracy"]
        .agg(ens_mean="mean", ens_std="std")
        .reset_index()
    )
    ens_agg["ens_std"] = ens_agg["ens_std"].fillna(0)

    if "comp_mean_acc" in ens_filtered.columns:
        comp_agg = (
            ens_filtered.groupby("rho_numeric")["comp_mean_acc"]
            .agg(comp_mean="mean", comp_std="std")
            .reset_index()
        )
        comp_agg["comp_std"] = comp_agg["comp_std"].fillna(0)

# %% PLOT
_colors = _px.colors.qualitative.Plotly

all_rhos = sorted(
    set(
        ml_agg["rho_numeric"].dropna().tolist()
        + (ens_agg["rho_numeric"].dropna().tolist() if not ens_agg.empty else [])
        + (comp_agg["rho_numeric"].dropna().tolist() if comp_agg is not None else [])
    )
)
rho_to_pos = {v: i for i, v in enumerate(all_rhos)}
rho_labels = [str(int(v) if v == int(v) else v) for v in all_rhos]

fig = go.Figure()

for i, label in enumerate(sorted(ml_agg["line_label"].unique().tolist())):
    df = ml_agg[ml_agg["line_label"] == label].sort_values("rho_numeric")
    color = _colors[i % len(_colors)]
    fig.add_trace(
        go.Scatter(
            x=[rho_to_pos[v] for v in df["rho_numeric"].tolist()],
            y=df["acc_mean"].tolist(),
            name=label,
            mode="lines+markers",
            line=dict(color=color),
            marker=dict(color=color, size=6),
        )
    )

if not ens_agg.empty:
    ens_sorted = ens_agg.sort_values("rho_numeric")
    fig.add_trace(
        go.Scatter(
            x=[rho_to_pos[v] for v in ens_sorted["rho_numeric"].tolist()],
            y=ens_sorted["ens_mean"].tolist(),
            name=f"ensemble ({VOTE_METHOD})",
            mode="lines+markers",
            line=dict(color="black", dash="dash"),
            marker=dict(color="black", size=6),
        )
    )

if comp_agg is not None:
    comp_sorted = comp_agg.sort_values("rho_numeric")
    fig.add_trace(
        go.Scatter(
            x=[rho_to_pos[v] for v in comp_sorted["rho_numeric"].tolist()],
            y=comp_sorted["comp_mean"].tolist(),
            name="component mean",
            mode="lines+markers",
            line=dict(color="gray", dash="dot"),
            marker=dict(color="gray", size=6),
        )
    )

fig.update_layout(
    title=f"Metalearner accuracy vs ρ — {META_TYPE}",
    xaxis=dict(tickvals=list(range(len(all_rhos))), ticktext=rho_labels, title="ρ"),
    yaxis=dict(title="Accuracy"),
    legend=dict(orientation="v", x=1.02, xanchor="left"),
    template="plotly_white",
    height=600,
)
fig.show()
save_plot(fig, "analysis_profiles_comparison")
