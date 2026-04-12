# %%
import polars as pl

from src.config.notebook import setup_environment

cfg, experiment = setup_environment()

import notebooks.mlflow_helpers as mh
from notebooks.mlflow_helpers import save_plot

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
    merged = ensembles.join(
        cons, on="tags.component_set_hash", how="inner", suffix="_cons"
    )
    print("merged rows:", merged.height)
    display(merged.columns)

    ens_cons = merged[
        ["run_id", "run_id_cons", "tags.rho", "metrics.mean_rsa_correlation"]
    ]
    display(ens_cons.sort("tags.rho"))

# %% PLOT
import plotly.graph_objects as go

if "ens_cons" not in dir() or ens_cons is None:
    print("No data to plot.")
else:
    import pandas as pd

    agg = (
        ens_cons.with_columns(pl.col("tags.rho").cast(pl.Float64).alias("rho_numeric"))
        .to_pandas()
        .groupby("rho_numeric")["metrics.mean_rsa_correlation"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("rho_numeric")
    )
    agg["std"] = agg["std"].fillna(0)

    rho_vals = agg["rho_numeric"].tolist()
    rho_labels = [str(int(v) if v == int(v) else v) for v in rho_vals]
    x_pos = list(range(len(rho_vals)))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_pos,
            y=agg["mean"].tolist(),
            mode="lines+markers",
            marker=dict(size=8, color="steelblue"),
            line=dict(color="steelblue"),
            error_y=dict(
                type="data",
                array=agg["std"].tolist(),
                visible=True,
                color="steelblue",
                thickness=1.5,
            ),
            name="mean RSA correlation",
        )
    )
    fig.update_layout(
        title="Ensemble consistency (RSA correlation) vs ρ",
        xaxis=dict(tickvals=x_pos, ticktext=rho_labels, title="ρ"),
        yaxis=dict(title="Mean RSA correlation"),
        template="plotly_white",
    )
    fig.show()
    save_plot(fig, "ensemble_consistency")
