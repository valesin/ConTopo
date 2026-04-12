# %% [markdown]
# # Manual SIMPROF Confirmation (Streamlined)
#
# This notebook performs a single, coherent flow:
# 1. Connect to MLflow and load config.
# 2. Load and normalize run tables (models, inference, profiles, metalearners).
# 3. Link model → inference → profile for a target split/metric/rho.
# 4. Cache embeddings and profiles once; inspect first embedding/profile.
# 5. Reconstruct one adapter input row (embeddings + profile RDM) for a chosen sample.
# 6. Verify train-index assumption and compare with saved adapter_inputs artifacts.

# %% [markdown]
# ## 1) Setup

# %%
import numpy as np
import torch
import polars as pl
import mlflow

from src.config.notebook import setup_environment

cfg, exp = setup_environment()

import notebooks.mlflow_helpers as mh
from src.ensemble.selector import discover_ensembles
from src.mlflow_utils import get_inference_run, get_profile_run, component_set_hash
from src.data.loaders import get_split_labels

print("experiment:", exp.name)

# %% [markdown]
# ## 2) Load run tables once

# %%
models = mh.get_base_model_list(exp)
inf = mh.get_inference_list(exp)
profiles = mh.get_category_similarity_list(exp)
metal = mh.get_metalearner_list(exp)

print(
    "models:",
    models.shape,
    "| inference:",
    inf.shape,
    "| profiles:",
    profiles.shape,
    "| metalearners:",
    metal.shape,
)

# %% [markdown]
# ## 3) Normalize columns and build linked table (model → inference → profile)


# %%
# helpers
def _safe_col(df: pl.DataFrame, name: str) -> pl.Expr:
    return (
        pl.col(name).cast(pl.Utf8)
        if name in df.columns
        else pl.lit(None, dtype=pl.Utf8)
    )


# targets
target_rho = "0.0"
target_split = "test"
target_metric = "cosine"

# normalized tables
models_norm = (
    models.with_columns(
        [
            pl.coalesce(
                [_safe_col(models, "tags.rho"), _safe_col(models, "params.rho")]
            ).alias("rho"),
            pl.coalesce(
                [_safe_col(models, "tags.trial"), _safe_col(models, "params.trial")]
            ).alias("trial"),
            pl.coalesce(
                [
                    _safe_col(models, "tags.topology"),
                    _safe_col(models, "params.topology"),
                ]
            ).alias("topology"),
        ]
    )
    .rename({"run_id": "model_run_id"})
    .select(["model_run_id", "rho", "trial", "topology"])
)

inf_norm = (
    inf.with_columns(
        [
            pl.coalesce(
                [
                    _safe_col(inf, "tags.parent_run_id"),
                    _safe_col(inf, "params.parent_run_id"),
                    _safe_col(inf, "tags.trained_model_run_id"),
                    _safe_col(inf, "params.trained_model_run_id"),
                ]
            ).alias("model_run_id"),
            pl.coalesce(
                [_safe_col(inf, "tags.split"), _safe_col(inf, "params.split")]
            ).alias("split"),
        ]
    )
    .rename({"run_id": "inference_run_id"})
    .select(["inference_run_id", "model_run_id", "split"])
)

prof_norm = profiles.with_columns(
    [
        pl.coalesce(
            [
                _safe_col(profiles, "tags.parent_run_id"),
                _safe_col(profiles, "params.parent_run_id"),
            ]
        ).alias("model_run_id"),
        pl.coalesce(
            [
                _safe_col(profiles, "tags.inference_run_id"),
                _safe_col(profiles, "params.inference_run_id"),
            ]
        ).alias("inference_run_id"),
        pl.coalesce(
            [
                _safe_col(profiles, "tags.similarity_metric"),
                _safe_col(profiles, "params.similarity_metric"),
            ]
        ).alias("similarity_metric"),
        pl.coalesce(
            [_safe_col(profiles, "tags.split"), _safe_col(profiles, "params.split")]
        ).alias("split"),
    ]
).rename({"run_id": "profile_run_id"})

# filter + first profile per inference
inf_f = inf_norm.filter(pl.col("split") == target_split)

prof_f = prof_norm.filter(
    (pl.col("split") == target_split) & (pl.col("similarity_metric") == target_metric)
)
if "start_time" in prof_f.columns:
    prof_f = prof_f.sort("start_time")
prof_first = prof_f.group_by(
    ["model_run_id", "inference_run_id", "similarity_metric"]
).agg(
    pl.col("profile_run_id").first().alias("profile_run_id"),
    pl.col("split").first().alias("split"),
)

# linked table
linked = (
    models_norm.join(inf_f, on="model_run_id", how="inner")
    .join(prof_first, on=["model_run_id", "inference_run_id"], how="inner")
    .filter(pl.col("rho") == target_rho)
    .select(
        [
            "model_run_id",
            "inference_run_id",
            "profile_run_id",
            "rho",
            "trial",
            "topology",
            "similarity_metric",
            "split",
        ]
    )
)

print("linked rows:", linked.height)
display(linked.head(10))

# %% [markdown]
# ## 4) Cache embeddings/profiles once and inspect first examples

# %%
embeddings_cache = {}
profiles_cache = {}
failures = []

for rec in linked.to_dicts():
    inf_run_id = rec["inference_run_id"]
    prof_run_id = rec["profile_run_id"]
    try:
        if inf_run_id not in embeddings_cache:
            _, inf_tensors = mh.load_inference_results(
                inf_run_id, artifact_path="inference"
            )
            embeddings_cache[inf_run_id] = inf_tensors["embeddings"]
        if prof_run_id not in profiles_cache:
            _, prof_tensor = mh.load_profile_results(
                run_id=prof_run_id,
                split=target_split,
                similarity_metric=target_metric,
                artifact_path="profiles",
            )
            profiles_cache[prof_run_id] = prof_tensor
    except Exception as exc:
        failures.append(
            {
                "inference_run_id": inf_run_id,
                "profile_run_id": prof_run_id,
                "error": str(exc),
            }
        )

print(
    "embeddings cached:",
    len(embeddings_cache),
    "| profiles cached:",
    len(profiles_cache),
)
if failures:
    display(pl.DataFrame(failures).head(5))
else:
    # show one example
    first_inf = next(iter(embeddings_cache))
    first_prof = next(iter(profiles_cache))
    print("example embedding shape:", embeddings_cache[first_inf][0].shape)
    print("example profile shape:", profiles_cache[first_prof][0].shape)

# %%
first_embedding = next(iter(embeddings_cache))
first_profile = next(iter(profiles_cache))
print(embeddings_cache[first_embedding][0])
profiles_cache[first_profile][0]

# %% [markdown]
# ## 5) One-pass reconstruction of an adapter input row (embeddings + profile RDM)

# %%
# config + ensemble selection
sample_idx = 0
ensemble_name = None

split = "test"
similarity_metric = target_metric

groups = discover_ensembles(cfg.mlflow.experiment_name)
if not groups:
    raise RuntimeError("No dynamic ensembles discovered.")
if ensemble_name is None:
    ensemble_name = sorted(groups.keys())[0]
run_ids = groups[ensemble_name]

print("ensemble:", ensemble_name, "| components:", len(run_ids))

# %% [markdown]
# ## 6) Verify train-index assumption and compare with saved adapter_inputs

# %%
# pick latest finished metalearner with embeddings+profiles and target rho/split
status_col = "status" if "status" in metal.columns else "attributes.status"
metal_f = metal
if status_col in metal.columns:
    metal_f = metal_f.filter(pl.col(status_col) == "FINISHED")

metal_f = metal_f.with_columns(
    [
        pl.coalesce(
            [
                (
                    pl.col("tags.feature_type").cast(pl.Utf8)
                    if "tags.feature_type" in metal_f.columns
                    else pl.lit(None, dtype=pl.Utf8)
                ),
                (
                    pl.col("params.feature_type").cast(pl.Utf8)
                    if "params.feature_type" in metal_f.columns
                    else pl.lit(None, dtype=pl.Utf8)
                ),
            ]
        ).alias("feature_type_sel"),
        pl.coalesce(
            [
                (
                    pl.col("tags.rho").cast(pl.Utf8)
                    if "tags.rho" in metal_f.columns
                    else pl.lit(None, dtype=pl.Utf8)
                ),
                (
                    pl.col("params.rho").cast(pl.Utf8)
                    if "params.rho" in metal_f.columns
                    else pl.lit(None, dtype=pl.Utf8)
                ),
            ]
        ).alias("rho_sel"),
        pl.coalesce(
            [
                (
                    pl.col("tags.split").cast(pl.Utf8)
                    if "tags.split" in metal_f.columns
                    else pl.lit(None, dtype=pl.Utf8)
                ),
                (
                    pl.col("params.split").cast(pl.Utf8)
                    if "params.split" in metal_f.columns
                    else pl.lit(None, dtype=pl.Utf8)
                ),
            ]
        ).alias("split_sel"),
        pl.coalesce(
            [
                (
                    pl.col("tags.component_set_hash").cast(pl.Utf8)
                    if "tags.component_set_hash" in metal_f.columns
                    else pl.lit(None, dtype=pl.Utf8)
                ),
            ]
        ).alias("component_set_hash_sel"),
    ]
)

metal_f = metal_f.filter(
    pl.col("feature_type_sel").str.contains("embeddings")
    & pl.col("feature_type_sel").str.contains("profiles")
    & (pl.col("rho_sel") == target_rho)
    & (pl.col("split_sel") == target_split)
)

if metal_f.is_empty():
    raise RuntimeError(
        "No FINISHED metalearner run matches embeddings+profiles + target rho/split."
    )

sort_col = "end_time" if "end_time" in metal_f.columns else "start_time"
metal_ref = metal_f.sort(sort_col, descending=True).head(1)
meta_run_id = metal_ref["run_id"][0]


def _first_val(df: pl.DataFrame, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            v = df[c][0]
            if v is not None:
                return v
    return default


meta_split_seed = int(_first_val(metal_ref, ["params.meta_split_seed"], 0))
meta_split_train = float(_first_val(metal_ref, ["params.meta_split_train"], 0.6))
meta_split_val = float(_first_val(metal_ref, ["params.meta_split_val"], 0.2))
meta_similarity_metric = str(
    _first_val(
        metal_ref,
        ["params.similarity_metric", "tags.similarity_metric"],
        similarity_metric,
    )
)
meta_component_set_hash = str(_first_val(metal_ref, ["tags.component_set_hash"], ""))

print("meta_run_id:", meta_run_id)
print(
    "meta_split_seed:",
    meta_split_seed,
    "| train:",
    meta_split_train,
    "| val:",
    meta_split_val,
)
print("meta_similarity_metric:", meta_similarity_metric)
print("component_set_hash:", meta_component_set_hash)

# %%
# permutation-based split to find first train index
N_total = len(get_split_labels(cfg, "test"))
rng = np.random.default_rng(meta_split_seed)
indices = rng.permutation(N_total)

n_train = int(N_total * meta_split_train)
n_val = int(N_total * meta_split_val)
train_idx = indices[:n_train]
val_idx = indices[n_train : n_train + n_val]
holdout_idx = indices[n_train + n_val :]

first_train_idx = int(train_idx[0])
print("N_total:", N_total)
print(
    "first_train_idx:",
    first_train_idx,
    "| assumption (first train == original index 0):",
    first_train_idx == 0,
)

# %%
# resolve component runs from component_set_hash
groups_all = discover_ensembles(cfg.mlflow.experiment_name)
match = []
for ens_name_k, ids_k in groups_all.items():
    if component_set_hash(ids_k) == meta_component_set_hash:
        match.append((ens_name_k, ids_k))
if not match:
    raise RuntimeError("Could not resolve component run_ids from component_set_hash")
ens_name_ref, run_ids_ref = match[0]
print("resolved ensemble:", ens_name_ref, "| components:", len(run_ids_ref))

# %%
# load adapter_inputs artifact (if present)
artifact_first_train_row = None
artifact_first_train_original_idx = None
artifact_inputs_available = False

try:
    infos = mlflow.artifacts.list_artifacts(run_id=meta_run_id, artifact_path="inputs")
    paths = [i.path for i in infos]
    npz_candidates = [p for p in paths if p.endswith(".npz") and "adapter_inputs_" in p]
    if npz_candidates:
        npz_local = mlflow.artifacts.download_artifacts(
            run_id=meta_run_id, artifact_path=npz_candidates[0]
        )
        with np.load(npz_local) as d:
            X_train_art = d["X_train"]
            train_idx_art = d["train_idx"]
            artifact_first_train_row = X_train_art[0]
            artifact_first_train_original_idx = int(train_idx_art[0])
            artifact_inputs_available = True
    print("artifact_inputs_available:", artifact_inputs_available)
    if artifact_inputs_available:
        print("artifact_first_train_original_idx:", artifact_first_train_original_idx)
except Exception as exc:
    print("Could not load adapter_inputs artifacts:", exc)

# %%
# manual reconstruction for metalearner's first train sample
# TODO: get_embedding and get_profile are not defined — implement or remove this cell
manual_idx = first_train_idx
labels_test = get_split_labels(cfg, "test")
y_manual = int(labels_test[manual_idx].item())

base_vecs_ref = []
profile_vecs_ref = []

for run_id_ref in run_ids_ref:
    base_vecs_ref.append(get_embedding(run_id_ref, "test", manual_idx))
    prof_vec_ref, _ = get_profile(
        run_id_ref, meta_similarity_metric, "test", manual_idx
    )
    profile_vecs_ref.append(prof_vec_ref)

x_base_ref = torch.cat(base_vecs_ref, dim=0)
P_ref = torch.stack(profile_vecs_ref, dim=0)
M_ref, C_ref = P_ref.shape
mask_ref = torch.ones(C_ref, dtype=torch.bool)
mask_ref[y_manual] = False
P_masked_ref = P_ref[:, mask_ref]
Pc_ref = P_masked_ref - P_masked_ref.mean(dim=1, keepdim=True)
P_norm_ref = Pc_ref.norm(dim=1, keepdim=True).clamp_min(1e-8)
Pn_ref = Pc_ref / P_norm_ref
corr_ref = Pn_ref @ Pn_ref.T
rdm_ref = 1.0 - corr_ref
tri_ref = torch.triu_indices(M_ref, M_ref, offset=1)
S_ref = rdm_ref[tri_ref[0], tri_ref[1]]
manual_first_train_row = torch.cat([x_base_ref, S_ref], dim=0).detach().cpu().numpy()

print("manual_idx:", manual_idx)
print("manual_first_train_row shape:", manual_first_train_row.shape)

# %%
# compare manual vs artifact
print("assumption (first train == test index 0):", first_train_idx == 0)
if artifact_inputs_available:
    print(
        "artifact_first_train_original_idx == first_train_idx:",
        artifact_first_train_original_idx == first_train_idx,
    )
    if manual_first_train_row.shape != artifact_first_train_row.shape:
        print(
            "shape mismatch:",
            manual_first_train_row.shape,
            artifact_first_train_row.shape,
        )
    else:
        diff = np.abs(manual_first_train_row - artifact_first_train_row)
        print("max_abs_diff:", float(diff.max()))
        print("mean_abs_diff:", float(diff.mean()))
        print(
            "allclose(atol=1e-6, rtol=1e-5):",
            bool(
                np.allclose(
                    manual_first_train_row,
                    artifact_first_train_row,
                    atol=1e-6,
                    rtol=1e-5,
                )
            ),
        )
else:
    print(
        "No adapter_inputs artifact available yet — re-run adapters after logging patch, then rerun this section."
    )
