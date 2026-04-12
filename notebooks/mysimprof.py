# %% [markdown]
# # MySIMPROF
# Setup the mlflow connection

# %%
import numpy as np
import torch
import torch.nn.functional as F
import polars as pl

from src.config.notebook import setup_environment

cfg, exp = setup_environment()

import notebooks.mlflow_helpers as mh
from src.data.anchors import get_or_create_anchors
from src.data.loaders import get_split_labels
from src.config.paths import get_cache_dir

print("experiment:", exp.name)

# %% [markdown]
# Select a specific metalearner

# %%
meta = mh.get_metalearner_list(exp)
print(meta.columns)
# Actually I don't need to merge them with the ensembles, because the csv is in the metalearners too
meta = meta.with_columns(
    pl.col("params.component_run_ids_csv").str.split(",").alias("components_list")
)

meta_one = meta.filter(
    (pl.col("tags.rho") == "0.0")
    & (pl.col("tags.split") == "test")
    & (pl.col("tags.feature_type") == "embeddings+profiles")
    & (pl.col("tags.similarity_metric") == "cosine")
    & (pl.col("tags.behaviour") == "meta_mlp_2")
)
meta_one

# %% [markdown]
# Retrieve the first metalearner's input, and the corresponding raw embeddings from the component models.
# It is pointless to compare now the models' output and the metalearner input, since the concatenated
# features are normalised including also the rdm, not calculated yet at this point

# %%
_, adapter_data = mh.load_adapter_inputs(meta_one["run_id"].to_list()[0])

# The original dataset index of X_train[0]
original_idx = int(adapter_data["train_idx"][0])
print(f"X_train[0] corresponds to original dataset index: {original_idx}")

components = meta_one["components_list"].to_list()[0]
components

# # Now load the raw inference embeddings for any component model and compare
embs = [
    mh.load_inference_results_from_model_run_id(exp, c)[1]["embeddings"][original_idx]
    for c in components
]
for emb in embs:
    print(f"Raw embedding[:5]: {emb[:5]}")

# %% [markdown]
# Retrieve the anchors using the same params as the ones used for metalearners

# %%
split = "test"
labels = get_split_labels(cfg, split)

sel = cfg.pipeline.anchors
anchors = get_or_create_anchors(
    labels=labels,
    source_split=sel.source_split,
    per_class=sel.per_class,
    strategy=sel.strategy,
    order_by=sel.order_by,
    num_classes=cfg.dataset.num_classes,
    artifacts_root=str(get_cache_dir(cfg)),
    dataset_name=cfg.dataset.name,
)

anchor_indices = anchors["anchor_indices"]
print(f"Loaded {len(anchor_indices)} anchors")
assert len(anchor_indices) == len(set(anchor_indices))

# %% [markdown]
# Calculate the class-averaged cosine similarity between the selected image and the anchors

# %%
class_similarities = {}

for c in components:
    # 1. Fetch
    _, tensors = mh.load_inference_results_from_model_run_id(exp, c)
    # N num of embeddings
    # D length of embeddings
    # K num of anchors
    # C num of classes
    embeddings = torch.from_numpy(tensors["embeddings"])  # [N, D]

    # 2. Slice anchors
    current_anchors = embeddings[anchors["anchor_indices"]]  # Shape: [K, D]

    # 3. Target vector
    first = embeddings[original_idx]  # Shape: [D]

    # 3b. Get cosine similarity
    similarities = F.cosine_similarity(
        current_anchors, first.unsqueeze(0), dim=1
    )  # Shape: [K]

    # 4. Group anchors by class
    per_class = anchors["spec"]["per_class"]
    num_classes = anchors["spec"]["num_classes"]
    anchors_by_class = similarities.view(num_classes, per_class)  # Shape: [C, K/C]

    # 5. For a given image (in this case the first), average over them and save the result
    class_similarities[c] = anchors_by_class.mean(dim=1)

class_similarities

# %% [markdown]
# Calculate RDMs across

# %%
from scipy.stats import pearsonr

# Mask out the true class from each model's profile
true_label = int(adapter_data["y_train"][0])
masked_similarities = {}
for c, profile in class_similarities.items():
    masked_similarities[c] = torch.cat(
        [profile[:true_label], profile[true_label + 1 :]]
    )  # [C-1]

# Build RDM
component_ids = list(masked_similarities.keys())
M = len(component_ids)
rdm = np.zeros((M, M))

for i in range(M):
    for j in range(M):
        if i == j:
            rdm[i, j] = 0.0
        else:
            profile_i = masked_similarities[component_ids[i]].numpy()
            profile_j = masked_similarities[component_ids[j]].numpy()
            corr, _ = pearsonr(profile_i, profile_j)
            rdm[i, j] = 1.0 - corr

print(f"RDM Shape: {rdm.shape}")
print(rdm)

# Print first 5 elements of each model's masked profile + the full profile value
for c in component_ids:
    prof = masked_similarities[c]
    # Mean-center and normalise (like Pearson does internally)
    centered = prof - prof.mean()
    normed = centered / centered.norm().clamp_min(1e-8)
    print(f"\nModel {c[:8]}...")
    print(f"  Masked profile[:5]:     {prof[:5].numpy()}")
    print(f"  Normalised profile[:5]: {normed[:5].numpy()}")

# Extract upper triangle
upper_triangle_indices = np.triu_indices_from(rdm, k=1)
simcat = rdm[upper_triangle_indices]
print(f"\nExtracted {len(simcat)} unique pairwise dissimilarities:")
print(simcat)

# %%
# Convert embeddings to numpy arrays and concatenate
raw_base = np.concatenate([e for e in embs])  # Shape: [512]

# Glue the RDM profile feature at the end
full_raw_input = np.concatenate([raw_base, simcat])  # Shape: [513]

print(f"Reconstructed full raw input (shape {full_raw_input.shape})")

# %%
# 1. Grab the training standardisation stats
mean = adapter_data["standardize_mean"][0]  # Shape: [513]
std = adapter_data["standardize_std"][0]  # Shape: [513]

# 2. Standardise our manually rebuilt input
manual_standardised = (full_raw_input - mean) / (std + 1e-6)

# 3. Fetch what script 05 genuinely stored for X_train[0]
actual_xtrain_0 = adapter_data["X_train"][0]

# 4. Prove they match (np.allclose safely ignores tiny float32 precision rounding)
is_match = np.allclose(manual_standardised, actual_xtrain_0, atol=1e-4)

print(f"Manual input perfectly matches MLflow X_train[0]: {is_match}")

if not is_match:
    diff = np.abs(manual_standardised - actual_xtrain_0)
    print(f"\nMax difference found: {diff.max():.8f}")
