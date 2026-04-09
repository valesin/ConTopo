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
from src.config.notebook import setup_environment

cfg, experiment = setup_environment()

from notebooks.mlflow.mlflow_helpers import (
    get_metalearner_results,
    get_ensemble_results,
    save_plot,
)

print(f"Experiment: {experiment.name}")

# %% LOAD
ml_df = get_metalearner_results(experiment.name)
ens_df = get_ensemble_results(experiment.name, split=SPLIT)

# %% INSPECT
print(f"Metalearner runs:  {len(ml_df)}")
print(f"Ensemble runs:     {len(ens_df)}")

if not ml_df.empty:
    print(
        "\nUnique rho values:     ",
        sorted(ml_df["rho_numeric"].dropna().unique().tolist()),
    )
    print(
        "Unique feature_type:   ",
        sorted(ml_df["feature_type"].dropna().unique().tolist()),
    )
    print(
        "Unique meta_type:      ", sorted(ml_df["meta_type"].dropna().unique().tolist())
    )
    if "profile_mask" in ml_df.columns:
        print(
            "Unique profile_mask:   ",
            sorted(ml_df["profile_mask"].dropna().unique().tolist()),
        )

if not ens_df.empty:
    print(
        "\nUnique vote_method:    ",
        sorted(ens_df["vote_method"].dropna().unique().tolist()),
    )

# %% FILTER
import pandas as pd

ml_filtered = ml_df.copy()

# Drop old runs that predate the profile_mask parameter
if "profile_mask" in ml_filtered.columns:
    ml_filtered = ml_filtered[ml_filtered["profile_mask"] != "N/A"]

# Exclude leaky true_class mask
if "profile_mask" in ml_filtered.columns:
    ml_filtered = ml_filtered[ml_filtered["profile_mask"] != "true_class"]

ml_filtered = ml_filtered[ml_filtered["meta_type"] == META_TYPE]
ml_filtered = ml_filtered[ml_filtered["feature_type"].isin(FEATURE_TYPES)]

if SIM_METRIC is not None and "similarity_metric" in ml_filtered.columns:
    ml_filtered = ml_filtered[ml_filtered["similarity_metric"] == SIM_METRIC]

ens_filtered = (
    ens_df[ens_df["vote_method"] == VOTE_METHOD] if not ens_df.empty else ens_df
)

print(f"Metalearner rows after filter: {len(ml_filtered)}")
print(f"Ensemble rows after filter:    {len(ens_filtered)}")

if not ml_filtered.empty:
    group_cols = ["meta_type", "feature_type"]
    if "profile_mask" in ml_filtered.columns:
        group_cols.append("profile_mask")
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
GROUP_COLS = ["rho_numeric", "meta_type", "feature_type"]
if not ml_filtered.empty and "profile_mask" in ml_filtered.columns:
    GROUP_COLS.append("profile_mask")

if not ml_filtered.empty:
    ml_agg = (
        ml_filtered.groupby(GROUP_COLS)["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "acc_mean", "std": "acc_std"})
    )
    ml_agg["acc_std"] = ml_agg["acc_std"].fillna(0)
else:
    ml_agg = pd.DataFrame(columns=GROUP_COLS + ["acc_mean", "acc_std"])

if not ens_filtered.empty:
    ens_agg = (
        ens_filtered.groupby("rho_numeric")["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "ens_mean", "std": "ens_std"})
    )
    ens_agg["ens_std"] = ens_agg["ens_std"].fillna(0)

    comp_agg = None
    if "comp_mean_acc" in ens_filtered.columns:
        comp_agg = (
            ens_filtered.groupby("rho_numeric")["comp_mean_acc"]
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

_colors = _px.colors.qualitative.Plotly

pairs = (
    sorted(ml_agg.groupby(["meta_type", "feature_type"]).groups.keys())
    if not ml_agg.empty
    else []
)
profile_masks = (
    sorted(ml_agg["profile_mask"].dropna().unique().tolist())
    if not ml_agg.empty and "profile_mask" in ml_agg.columns
    else []
)

# Build equally-spaced positions from all rho values across ml_agg + baselines
all_rhos = sorted(
    set(
        ml_agg["rho_numeric"].dropna().tolist()
        + (ens_agg["rho_numeric"].dropna().tolist() if not ens_agg.empty else [])
        + (comp_agg["rho_numeric"].dropna().tolist() if comp_agg is not None else [])
    )
)
rho_to_pos = {v: i for i, v in enumerate(all_rhos)}
rho_labels = [str(int(v) if v == int(v) else v) for v in all_rhos]

# Jitter: symmetric additive offsets on integer positions so error bars don't stack
n_masks = len(profile_masks)
jitter_offsets = [(i - (n_masks - 1) / 2) * 0.08 for i in range(n_masks)]

fig = go.Figure()

for pair_idx, (meta_type, feature_type) in enumerate(pairs):
    group_name = feature_type
    pair_df = ml_agg[
        (ml_agg["meta_type"] == meta_type) & (ml_agg["feature_type"] == feature_type)
    ]
    is_first_pair = pair_idx == 0

    for mask_idx, pm in enumerate(profile_masks):
        mask_df = pair_df[pair_df["profile_mask"] == pm].sort_values("rho_numeric")
        if mask_df.empty:
            continue
        color = _colors[mask_idx % len(_colors)]
        x = [
            rho_to_pos[v] + jitter_offsets[mask_idx]
            for v in mask_df["rho_numeric"].tolist()
        ]

        fig.add_trace(
            go.Scatter(
                x=x,
                y=mask_df["acc_mean"].tolist(),
                name=pm,
                legendgroup=group_name,
                legendgrouptitle_text=group_name if mask_idx == 0 else None,
                showlegend=True,
                visible=True if is_first_pair else "legendonly",
                mode="lines+markers",
                line=dict(color=color),
                marker=dict(color=color, size=6),
                error_y=dict(
                    type="data",
                    array=mask_df["acc_std"].tolist(),
                    visible=True,
                    color=color,
                    thickness=1.5,
                ),
            )
        )

# Baselines — always visible, own legend group
if not ens_agg.empty:
    ens_sorted = ens_agg.sort_values("rho_numeric")
    fig.add_trace(
        go.Scatter(
            x=[rho_to_pos[v] for v in ens_sorted["rho_numeric"].tolist()],
            y=ens_sorted["ens_mean"].tolist(),
            name=f"ensemble ({VOTE_METHOD})",
            legendgroup="baselines",
            legendgrouptitle_text="baselines",
            showlegend=True,
            visible=True,
            mode="lines+markers",
            line=dict(color="black", dash="dash"),
            marker=dict(color="black", size=6),
            error_y=dict(
                type="data",
                array=ens_sorted["ens_std"].tolist(),
                visible=True,
                color="black",
                thickness=1.5,
            ),
        )
    )

if comp_agg is not None:
    comp_sorted = comp_agg.sort_values("rho_numeric")
    fig.add_trace(
        go.Scatter(
            x=[rho_to_pos[v] for v in comp_sorted["rho_numeric"].tolist()],
            y=comp_sorted["comp_mean"].tolist(),
            name="component mean",
            legendgroup="baselines",
            showlegend=True,
            visible=True,
            mode="lines+markers",
            line=dict(color="gray", dash="dot"),
            marker=dict(color="gray", size=6),
            error_y=dict(
                type="data",
                array=comp_sorted["comp_std"].tolist(),
                visible=True,
                color="gray",
                thickness=1.5,
            ),
        )
    )

fig.update_layout(
    title=f"Metalearner accuracy vs ρ — {META_TYPE} (non-leaky conditions only)",
    xaxis=dict(
        tickvals=list(range(len(all_rhos))),
        ticktext=rho_labels,
        title="ρ",
    ),
    yaxis=dict(title="Accuracy"),
    legend=dict(orientation="v", x=1.02, xanchor="left", groupclick="toggleitem"),
    template="plotly_white",
    height=600,
)
fig.show()
save_plot(fig, "analysis_profiles_comparison")
