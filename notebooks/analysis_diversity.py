# %% [markdown]
# # Diversity analysis vs ρ
#
# Analyses whether increasing the topographic regularisation weight ρ leads to
# higher ensemble diversity. Five pairwise diversity metrics are examined:
# q_statistic, disagreement, double_fault, correlation, interrater_agreement.

# %% CONFIG
SPLIT = "test"
NUM_COMPONENTS = "10"
DIVERSITY_METRICS = [
    "q_statistic",
    "disagreement",
    "double_fault",
    "correlation",
    "interrater_agreement",
]

# Direction: True = higher value means more diverse, False = lower value means more diverse
METRIC_HIGHER_IS_MORE_DIVERSE = {
    "q_statistic": False,  # 0 = independent, 1 = fully correlated
    "disagreement": True,  # fraction of samples where classifiers disagree
    "double_fault": False,  # fraction where both classifiers are wrong together
    "correlation": False,  # 0 = independent, 1 = fully correlated
    "interrater_agreement": False,  # kappa-like: 0 = chance agreement, 1 = full agreement
}

# %% SETUP
from src.config.notebook import setup_environment

cfg, experiment = setup_environment()

import notebooks.mlflow.mlflow_helpers as mh
from notebooks.mlflow.mlflow_helpers import save_plot

print(f"Experiment: {experiment.name}")

# %% LOAD
import polars as pl

div_df = mh.get_diversity_list(experiment)
ens_df = mh.get_ensemble_list(experiment)

# %% INSPECT
print(f"Diversity runs:  {div_df.height}")
print(f"Ensemble runs:   {ens_df.height}")

if not div_df.is_empty() and "params.diversity_metric" in div_df.columns:
    print(
        "\nUnique diversity metrics:",
        sorted(div_df["params.diversity_metric"].drop_nulls().unique().to_list()),
    )

print("\nMetric directions:")
for m, higher in METRIC_HIGHER_IS_MORE_DIVERSE.items():
    direction = "higher → more diverse" if higher else "lower  → more diverse"
    print(f"  {m:<26} {direction}")

if not ens_df.is_empty() and "tags.rho" in ens_df.columns:
    print(
        "Unique rho values:       ",
        sorted(ens_df["tags.rho"].drop_nulls().unique().to_list()),
    )

# %% FILTER + JOIN
ens_filtered = ens_df.filter(
    (pl.col("params.method") == "soft")
    & (pl.col("params.num_components") == NUM_COMPONENTS)
).select(["tags.component_set_hash", "tags.rho"])

print(f"\nEnsembles after filter: {ens_filtered.height}")

merged = div_df.join(ens_filtered, on="tags.component_set_hash", how="inner")
print(f"Diversity rows after join: {merged.height}")

if merged.is_empty():
    raise RuntimeError(
        "No diversity runs matched the filtered ensembles. Check component_set_hash alignment."
    )

# Each diversity run has params.diversity_metric = 'q_statistic' etc.
# and metrics.q_statistic = <value>. Coalesce to a single 'value' column.
available_metrics = [m for m in DIVERSITY_METRICS if f"metrics.{m}" in merged.columns]
missing = set(DIVERSITY_METRICS) - set(available_metrics)
if missing:
    print(f"Warning: metrics not found in data: {missing}")

coalesce_expr = pl.coalesce(
    [
        pl.when(pl.col("params.diversity_metric") == m).then(pl.col(f"metrics.{m}"))
        for m in available_metrics
    ]
)

plot_df = (
    merged.with_columns(
        [
            coalesce_expr.alias("value"),
            pl.col("tags.rho").cast(pl.Float64).alias("rho_numeric"),
        ]
    )
    .filter(pl.col("params.diversity_metric").is_in(available_metrics))
    .select(["rho_numeric", "params.diversity_metric", "value"])
    .sort(["params.diversity_metric", "rho_numeric"])
)

print(f"\nPlot rows: {plot_df.height}")
print(plot_df)

# %% PLOT 1 — All metrics on one plot
import plotly.graph_objects as go
import plotly.express as _px

_colors = _px.colors.qualitative.Plotly

rho_vals = sorted(plot_df["rho_numeric"].unique().to_list())
rho_labels = [str(int(v) if v == int(v) else v) for v in rho_vals]
rho_to_pos = {v: i for i, v in enumerate(rho_vals)}

fig1 = go.Figure()

for i, metric in enumerate(available_metrics):
    grp = plot_df.filter(pl.col("params.diversity_metric") == metric).sort(
        "rho_numeric"
    )
    x = [rho_to_pos[v] for v in grp["rho_numeric"].to_list()]
    fig1.add_trace(
        go.Scatter(
            x=x,
            y=grp["value"].to_list(),
            name=metric,
            mode="lines+markers",
            line=dict(color=_colors[i % len(_colors)]),
            marker=dict(color=_colors[i % len(_colors)], size=8),
        )
    )

fig1.update_layout(
    title="Diversity metrics vs ρ",
    xaxis=dict(tickvals=list(range(len(rho_vals))), ticktext=rho_labels, title="ρ"),
    yaxis=dict(title="Diversity value"),
    legend=dict(orientation="v", x=1.02, xanchor="left"),
    template="plotly_white",
)
fig1.show()
save_plot(fig1, "diversity_all_metrics")

# %% PLOT 2 — Subplots, one per metric
from plotly.subplots import make_subplots

n_metrics = len(available_metrics)


def _metric_label(m):
    arrow = (
        "↑ more diverse"
        if METRIC_HIGHER_IS_MORE_DIVERSE.get(m, True)
        else "↓ more diverse"
    )
    return f"{m}  ({arrow})"


fig2 = make_subplots(
    rows=n_metrics,
    cols=1,
    subplot_titles=[_metric_label(m) for m in available_metrics],
    shared_xaxes=True,
    vertical_spacing=0.06,
)

for i, metric in enumerate(available_metrics):
    grp = plot_df.filter(pl.col("params.diversity_metric") == metric).sort(
        "rho_numeric"
    )
    x = [rho_to_pos[v] for v in grp["rho_numeric"].to_list()]
    color = _colors[i % len(_colors)]
    fig2.add_trace(
        go.Scatter(
            x=x,
            y=grp["value"].to_list(),
            name=metric,
            showlegend=False,
            mode="lines+markers",
            line=dict(color=color),
            marker=dict(color=color, size=8),
        ),
        row=i + 1,
        col=1,
    )
    fig2.update_yaxes(title_text=metric, row=i + 1, col=1)

fig2.update_xaxes(
    tickvals=list(range(len(rho_vals))),
    ticktext=rho_labels,
    title_text="ρ",
    row=n_metrics,
    col=1,
)
fig2.update_layout(
    title="Diversity metrics vs ρ (per metric)",
    height=220 * n_metrics,
    template="plotly_white",
)
fig2.show()
save_plot(fig2, "diversity_subplots")

# %% PLOT 3 — Spearman correlation table
from scipy import stats

corr_rows = []
for metric in available_metrics:
    grp = plot_df.filter(pl.col("params.diversity_metric") == metric).sort(
        "rho_numeric"
    )
    rho_numeric = grp["rho_numeric"].to_list()
    values = grp["value"].to_list()
    if len(rho_numeric) >= 3:
        r, p = stats.spearmanr(rho_numeric, values)
    else:
        r, p = float("nan"), float("nan")
    corr_rows.append({"metric": metric, "spearman_r": r, "p_value": p})

fig3 = go.Figure(
    go.Table(
        header=dict(
            values=["metric", "direction", "Spearman r", "p-value"],
            align="left",
            font=dict(size=13),
        ),
        cells=dict(
            values=[
                [row["metric"] for row in corr_rows],
                [
                    (
                        "↑ more diverse"
                        if METRIC_HIGHER_IS_MORE_DIVERSE.get(row["metric"], True)
                        else "↓ more diverse"
                    )
                    for row in corr_rows
                ],
                [f"{row['spearman_r']:.3f}" for row in corr_rows],
                [f"{row['p_value']:.3f}" for row in corr_rows],
            ],
            align="left",
        ),
    )
)
fig3.update_layout(
    title="Spearman correlation between ρ and diversity metrics",
    margin=dict(t=50, b=10, l=10, r=10),
    height=60 + 35 * len(corr_rows),
)
fig3.show()
save_plot(fig3, "diversity_correlation")
