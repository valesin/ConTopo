# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import mlflow
from src.mlflow_utils import get_ensemble_results, get_metalearner_results

# %%
EXPERIMENT_NAME = "contopo"
SPLIT = "test"

# %%
# ── MLflow connection ──
# Handle different execution contexts (root vs notebooks/)
tracking_uri = "sqlite:///outputs/mlflow.db"
if not os.path.exists("outputs"):
    tracking_uri = "sqlite:///../outputs/mlflow.db"

mlflow.set_tracking_uri(tracking_uri)

# %%
ens_df = get_ensemble_results(EXPERIMENT_NAME, split=SPLIT)
meta_df = get_metalearner_results(EXPERIMENT_NAME, split=SPLIT)
results_df = pd.concat([ens_df, meta_df], ignore_index=True)

if results_df.empty:
    print(
        f"⚠️ No behaviour runs found for experiment '{EXPERIMENT_NAME}' and split '{SPLIT}'."
    )
else:
    # We filter only for runs that have a valid numeric rho for plotting
    clean_df = results_df.dropna(subset=["rho_numeric"]).copy()

    print(f"Total behaviour runs: {len(results_df)}")
    print(
        f"Valid rho values for plotting: {sorted(clean_df['rho_numeric'].unique().tolist()) if not clean_df.empty else 'None'}"
    )

# %%
# %%
if results_df.empty:
    print("⚠️ No data available.")
else:
    behaviours = sorted(results_df["behaviour"].unique())

    for behaviour in behaviours:
        print(f"\n## Behaviour: {behaviour}")

        # Get subset and drop empty columns for this behaviour
        subset = results_df[results_df["behaviour"] == behaviour].copy()

        # Identify columns that are all NaN or "-" for this behaviour to keep it clean
        # but keep core ones like rho, accuracy
        cols_to_check = [
            "ensemble_name",
            "feature_type",
            "similarity_metric",
            "split_seed",
        ]
        cols_to_keep = ["rho_numeric", "accuracy"]
        for c in cols_to_check:
            if c in subset.columns and not subset[c].isna().all():
                cols_to_keep.insert(-1, c)  # Insert before accuracy

        # Display the table
        display_df = subset[cols_to_keep].sort_values(["rho_numeric"])

        try:
            from IPython.display import display, Markdown

            # Create a nice styled view
            styled = (
                display_df.style.background_gradient(
                    cmap="viridis", subset=["accuracy"]
                )
                .format({"accuracy": "{:.4f}", "rho_numeric": "{:.2f}"})
                .set_table_attributes('style="width: 100%; border-collapse: collapse;"')
            )

            display(styled)
        except ImportError:
            print(display_df.to_string(index=False))
