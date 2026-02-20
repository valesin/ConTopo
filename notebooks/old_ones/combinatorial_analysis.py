# %% [markdown]
# # Combinatorial Diversity vs. Performance Analysis
#
# Loads all ensemble results from `save/ensembles/`, builds a tidy Polars
# DataFrame, and explores how diversity metrics correlate with ensemble accuracy.

# %%
import os
import json
import torch
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# %% [markdown]
# ## 1. Load all ensemble results

# %%
SAVE_DIR = os.path.join(os.path.dirname(os.getcwd()), "save", "ensembles_test")

# If running from repo root, use:
# SAVE_DIR = "save/ensembles_test"

index_path = os.path.join(SAVE_DIR, "index.json")
with open(index_path) as f:
    index = json.load(f)

print(f"Found {len(index)} ensembles in {SAVE_DIR}")

# %%
rows = []
for h, names in index.items():
    hash_dir = os.path.join(SAVE_DIR, h)
    diversity_path = os.path.join(hash_dir, "diversity.pt")
    if not os.path.exists(diversity_path):
        continue
    main_data = torch.load(diversity_path, weights_only=False)

    row = {
        "hash": h,
        "subset_size": main_data.get("subset_size", len(names)),
        "n_models": len(main_data.get("run_names", names)),
        "run_names": ", ".join(main_data.get("run_names", names)),
    }
    # Flatten accuracy dict
    for k, v in main_data.get("acc", {}).items():
        row[k] = round(v, 6)
    # Flatten diversity dict
    for k, v in main_data.get("diversity", {}).items():
        row[k] = round(v, 6) if np.isfinite(v) else None

    rows.append(row)

df = pl.DataFrame(rows)
print(f"DataFrame shape: {df.shape}")
df

# %% [markdown]
# ## 2. Full metrics table

# %%
# Identify metric columns
acc_cols = [c for c in df.columns if c.startswith("acc_")]
div_cols = [c for c in df.columns if c.startswith("div_")]

print(f"Accuracy columns: {acc_cols}")
print(f"Diversity columns: {div_cols}")

# %%
# Show full table sorted by acc_soft descending
df.select(["hash", "subset_size"] + acc_cols + div_cols).sort("acc_soft", descending=True)

# %% [markdown]
# ## 3. Correlation matrix (Pearson & Spearman)

# %%
def compute_correlations(df: pl.DataFrame, acc_cols: list, div_cols: list) -> pl.DataFrame:
    """Compute Pearson and Spearman correlations between every (acc, div) pair."""
    records = []
    for acc in acc_cols:
        for div in div_cols:
            # Drop nulls
            sub = df.select([acc, div]).drop_nulls()
            if len(sub) < 3:
                continue
            a = sub[acc].to_numpy()
            d = sub[div].to_numpy()

            pr, p_p = stats.pearsonr(a, d)
            sr, p_s = stats.spearmanr(a, d)
            records.append({
                "accuracy": acc,
                "diversity": div,
                "pearson_r": round(pr, 4),
                "pearson_p": round(p_p, 4),
                "spearman_r": round(sr, 4),
                "spearman_p": round(p_s, 4),
                "n": len(sub),
            })
    return pl.DataFrame(records)


corr_df = compute_correlations(df, acc_cols, div_cols)
corr_df.sort("pearson_r")

# %% [markdown]
# ## 4. Heatmap: Pearson correlation

# %%
pivot = corr_df.pivot(on="diversity", index="accuracy", values="pearson_r")
acc_labels = pivot["accuracy"].to_list()
div_labels = [c for c in pivot.columns if c != "accuracy"]
mat = pivot.select(div_labels).to_numpy().astype(float)

fig, ax = plt.subplots(figsize=(max(8, len(div_labels) * 1.2), max(4, len(acc_labels) * 0.8)))
im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(div_labels)))
ax.set_xticklabels([d.replace("div_", "") for d in div_labels], rotation=45, ha="right")
ax.set_yticks(range(len(acc_labels)))
ax.set_yticklabels([a.replace("acc_", "") for a in acc_labels])
# Annotate cells
for i in range(len(acc_labels)):
    for j in range(len(div_labels)):
        v = mat[i, j]
        color = "white" if abs(v) > 0.5 else "black"
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9, color=color)
plt.colorbar(im, label="Pearson r")
ax.set_title("Diversity → Accuracy Correlation (Pearson)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Scatter plots: each diversity metric vs. best accuracy

# %%
best_acc = acc_cols[0] if "acc_soft" not in acc_cols else "acc_soft"

n_div = len(div_cols)
ncols = min(3, n_div)
nrows = (n_div + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
axes_flat = axes.flatten()

for i, div in enumerate(div_cols):
    ax = axes_flat[i]
    sub = df.select([best_acc, div, "subset_size"]).drop_nulls()
    if len(sub) < 2:
        ax.set_title(div.replace("div_", ""))
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center")
        continue

    x = sub[div].to_numpy()
    y = sub[best_acc].to_numpy()
    sizes = sub["subset_size"].to_numpy()

    # Color by subset size
    scatter = ax.scatter(x, y, c=sizes, cmap="viridis", edgecolors="k", s=60, zorder=3)

    # Trend line
    if len(x) >= 2:
        z = np.polyfit(x, y, 1)
        xr = np.linspace(x.min(), x.max(), 50)
        ax.plot(xr, np.polyval(z, xr), "r--", alpha=0.7, linewidth=1.5)

    ax.set_xlabel(div.replace("div_", ""))
    ax.set_ylabel(best_acc.replace("acc_", "") + " accuracy")
    ax.set_title(div.replace("div_", ""))
    ax.grid(True, alpha=0.3)

# Hide unused axes
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

# Shared colorbar
fig.colorbar(scatter, ax=axes_flat[:n_div], label="Subset size k", shrink=0.6)
fig.suptitle(f"Diversity vs. {best_acc}", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Summary: which metric correlates most with accuracy?

# %%
summary = (
    corr_df
    .group_by("diversity")
    .agg([
        pl.col("pearson_r").mean().alias("avg_pearson_r"),
        pl.col("spearman_r").mean().alias("avg_spearman_r"),
        pl.col("pearson_r").abs().mean().alias("avg_abs_pearson"),
    ])
    .sort("avg_abs_pearson", descending=True)
)
print("Metrics ranked by average |Pearson r| across all accuracy measures:\n")
summary
