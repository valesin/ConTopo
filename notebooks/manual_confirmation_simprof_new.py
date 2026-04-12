# %% [markdown]
# # Manual SIMPROF Confirmation (New)
# This notebook reproduces manual SIMPROF confirmation in a standalone file and validates
# manifest/anchor behavior with MLflow lineage.

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
# ## 2) Load Base Model Runs and Inference Runs
# Fetch base model and inference run tables for lineage checks.

# %%
models = mh.get_base_model_list(exp)
inf = mh.get_inference_list(exp)

print("models:", models.shape)
print("inference:", inf.shape)

display(models.select(["run_id", "params.topology", "params.rho"]).head(5))
display(inf.select(["run_id", "tags.trained_model_run_id", "params.split"]).head(5))

# %% [markdown]
# ## 3) Join and Filter Runs with Polars
# Join inference to model runs and filter target subset (example: rho == 0.0).

# %%
merged = inf.join(
    models, left_on="tags.trained_model_run_id", right_on="run_id", how="inner"
)
merged = merged.filter(pl.col("params.rho") == "0.0")

print("merged rows:", merged.height)
display(
    merged.select(
        ["run_id", "tags.trained_model_run_id", "params.rho", "params.split"]
    ).head(10)
)

# %% [markdown]
# ## 4) Load Inference Artifacts and Extract Embeddings
# Load cached inference artifacts per run and store embeddings keyed by run_id.

# %%
embeddings_per_run = {}
failed_runs = []

for rid in merged["run_id"].to_list():
    try:
        _, inf_tensors = mh.load_inference_results(rid)
        embeddings_per_run[rid] = inf_tensors["embeddings"]
    except Exception as exc:
        failed_runs.append((rid, str(exc)))

print("loaded embeddings:", len(embeddings_per_run))
print("failed runs:", len(failed_runs))
if failed_runs:
    display(pl.DataFrame(failed_runs, schema=["run_id", "error"]).head(5))

# %% [markdown]
# ## 5) Modular retrieval of first embedding + first profile

# %%
# 5.1 Parameters + source tables
target_rho = "0.0"
target_split = "test"
target_metric = "cosine"

profiles = mh.get_category_similarity_list(exp)

print("models:", models.shape, "inference:", inf.shape, "profiles:", profiles.shape)


# %%
# 5.2 Normalize run tables
def _safe_col(df: pl.DataFrame, name: str) -> pl.Expr:
    return (
        pl.col(name).cast(pl.Utf8)
        if name in df.columns
        else pl.lit(None, dtype=pl.Utf8)
    )


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

print("normalized tables ready")

# %%
# 5.3 Build linked run table for target split/metric/rho
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

# %%
# 5.4 Retrieve first embedding + first profile per linked row
rows = []
failed = []

for rec in linked.to_dicts():
    inf_run_id = rec["inference_run_id"]
    prof_run_id = rec["profile_run_id"]
    try:
        _, inf_tensors = mh.load_inference_results(
            inf_run_id, artifact_path="inference"
        )
        if "embeddings" not in inf_tensors:
            raise KeyError(f"embeddings missing for inference run {inf_run_id}")
        first_embedding = inf_tensors["embeddings"][0]

        _, profile_tensor = mh.load_profile_results(
            run_id=prof_run_id,
            split=target_split,
            similarity_metric=target_metric,
            artifact_path="profiles",
        )
        if profile_tensor is None:
            raise FileNotFoundError(f"profile tensor missing for run {prof_run_id}")
        first_profile = profile_tensor[0].detach().cpu().numpy()

        rows.append(
            {
                **rec,
                "first_embedding": first_embedding,
                "first_profile": first_profile,
            }
        )
    except Exception as exc:
        failed.append({**rec, "error": str(exc)})

print("retrieved rows:", len(rows))
print("failed rows:", len(failed))

# %%
# 5.5 Inspect summary and failures
if rows:
    summary = pl.DataFrame(
        [
            {
                k: v
                for k, v in r.items()
                if k not in ("first_embedding", "first_profile")
            }
            for r in rows
        ]
    )
    display(summary.head(10))
    print("example embedding shape:", rows[0]["first_embedding"].shape)
    print("example profile shape:", rows[0]["first_profile"].shape)
else:
    print("No successful rows retrieved.")

if failed:
    display(pl.DataFrame(failed).head(10))

# %% [markdown]
# ## 6) Manual reconstruction of one adapter input row (embeddings + profiles only)
# Incremental flow: (1) config, (2) ensemble selection, (3) sample label,
# (4) load vectors, (5) build base part, (6) build profile-RDM part, (7) concatenate final input row.

# %%
# 6.1 Parameters (independent of cfg.adapter.feature_type)
sample_idx = 0
ensemble_name = None  # set string to pin a specific ensemble

split = "test"
similarity_metric = target_metric

print("split:", split)
print("similarity_metric:", similarity_metric)
print("manual path:", "embeddings + profiles")
print("sample_idx:", sample_idx)

# %%
# 6.2 Ensemble selection (same discovery path as training script)
groups = discover_ensembles(cfg.mlflow.experiment_name)
if not groups:
    raise RuntimeError("No dynamic ensembles discovered.")

if ensemble_name is None:
    ensemble_name = sorted(groups.keys())[0]
run_ids = groups[ensemble_name]

print("ensemble:", ensemble_name)
print("num component runs:", len(run_ids))
print("component run_ids (first 5):", run_ids[:5])

# %%
# 6.3 Label for this sample
labels_tensor = get_split_labels(cfg, split)
y_n = int(labels_tensor[sample_idx].item())
print("y_n (true class for sample_idx):", y_n)
print("num classes:", int(labels_tensor.max().item()) + 1)

# %%
# 6.4 Load one embedding row and one profile row per model
base_vecs = []
profile_vecs = []

for run_id in run_ids:
    inf_runs = get_inference_run(cfg.mlflow.experiment_name, run_id, split)
    if inf_runs.empty:
        raise RuntimeError(f"Missing '{split}' inference for {run_id}")
    inf_run_id = inf_runs.iloc[0].run_id

    _, inf_tensors = mh.load_inference_results(inf_run_id, artifact_path="inference")
    if "embeddings" not in inf_tensors:
        raise KeyError(f"Missing embeddings for inference run {inf_run_id}")
    emb_all = torch.from_numpy(inf_tensors["embeddings"]).float().cpu()
    base_vecs.append(emb_all[sample_idx])

    prof_runs = get_profile_run(
        cfg.mlflow.experiment_name, run_id, similarity_metric, split
    )
    if prof_runs.empty:
        raise RuntimeError(f"Missing '{similarity_metric}' profile for {run_id}")
    prof_run_id = prof_runs.iloc[0].run_id

    _, prof_tensor = mh.load_profile_results(
        run_id=prof_run_id,
        split=split,
        similarity_metric=similarity_metric,
        artifact_path="profiles",
    )
    if prof_tensor is None:
        raise RuntimeError(f"Missing profile tensor for run {prof_run_id}")
    profile_vecs.append(prof_tensor[sample_idx].float().cpu())

print("loaded vectors from", len(base_vecs), "models")
print("example embedding dim:", tuple(base_vecs[0].shape))
print("example profile dim:", tuple(profile_vecs[0].shape))

# %%
# 6.5 Build X_base part (concatenate embeddings across models)
x_base_n = torch.cat(base_vecs, dim=0)
print("x_base_n shape:", tuple(x_base_n.shape))
print("x_base_n[:10]:", np.round(x_base_n[:10].numpy(), 6))

# %%
# 6.6 Build profile-diversity part S_n (mask true class BEFORE correlation)
P_n = torch.stack(profile_vecs, dim=0)  # (M, C)
M, C = P_n.shape

class_mask = torch.ones(C, dtype=torch.bool)
class_mask[y_n] = False
P_masked_n = P_n[:, class_mask]  # (M, C-1)

Pc_n = P_masked_n - P_masked_n.mean(dim=1, keepdim=True)
P_norm_n = Pc_n.norm(dim=1, keepdim=True).clamp_min(1e-8)
Pn_n = Pc_n / P_norm_n

corr_n = Pn_n @ Pn_n.T
rdm_n = 1.0 - corr_n

idx = torch.triu_indices(M, M, offset=1)
S_n = rdm_n[idx[0], idx[1]]

print("P_n shape:", tuple(P_n.shape))
print("P_masked_n shape:", tuple(P_masked_n.shape))
print("S_n shape:", tuple(S_n.shape), "(expected", M * (M - 1) // 2, ")")

# %%
# 6.7 Final one-row adapter input point
manual_input_row = torch.cat([x_base_n, S_n], dim=0).detach().cpu().numpy()

print("manual_input_row shape:", manual_input_row.shape)
print("manual_input_row[:10]:", np.round(manual_input_row[:10], 6))

# %% [markdown]
# ## 7) Verify train-index assumption and compare with saved adapter inputs
# This section checks whether the first train row corresponds to original test index `0`,
# then compares `X_train[0]` from artifacts against a manual reconstruction for the same original index.

# %%
# 7.1 Pick reference metalearner run with strict constraints
metal = mh.get_metalearner_list(exp)
if metal.is_empty():
    raise RuntimeError("No metalearner runs found.")


def _first_val(df: pl.DataFrame, candidates: list[str], default=None):
    for c in candidates:
        if c in df.columns:
            v = df[c][0]
            if v is not None:
                return v
    return default


desired_rho = str(target_rho)
desired_split = "test"
expected_component_set_hash = None
if "linked" in globals() and isinstance(linked, pl.DataFrame) and linked.height > 0:
    expected_component_set_hash = component_set_hash(linked["model_run_id"].to_list())

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
    & (pl.col("rho_sel") == desired_rho)
    & (pl.col("split_sel") == desired_split)
)

if expected_component_set_hash is not None:
    metal_f = metal_f.filter(
        pl.col("component_set_hash_sel") == expected_component_set_hash
    )

if metal_f.is_empty():
    raise RuntimeError(
        "No FINISHED metalearner run matches embeddings+profiles + target rho (+ component_set_hash when available)."
    )

sort_col = "end_time" if "end_time" in metal_f.columns else "start_time"
metal_ref = metal_f.sort(sort_col, descending=True).head(1)
meta_run_id = metal_ref["run_id"][0]

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
print("selected feature_type:", _first_val(metal_ref, ["feature_type_sel"], "?"))
print(
    "selected rho:",
    _first_val(metal_ref, ["rho_sel"], "?"),
    "| desired_rho:",
    desired_rho,
)
print(
    "selected split:",
    _first_val(metal_ref, ["split_sel"], "?"),
    "| desired_split:",
    desired_split,
)
print(
    "meta_split_seed:",
    meta_split_seed,
    "| train:",
    meta_split_train,
    "| val:",
    meta_split_val,
)
print("meta_similarity_metric:", meta_similarity_metric)
print("component_set_hash (selected):", meta_component_set_hash)
if expected_component_set_hash is not None:
    print("component_set_hash (expected from linked):", expected_component_set_hash)

# %%
# 7.2 Verify assumption: first train row corresponds to original test index 0?
N_total = len(get_split_labels(cfg, "test"))
rng = np.random.default_rng(meta_split_seed)
indices = rng.permutation(N_total)

n_train = int(N_total * meta_split_train)
n_val = int(N_total * meta_split_val)
train_idx = indices[:n_train]
val_idx = indices[n_train : n_train + n_val]
holdout_idx = indices[n_train + n_val :]

first_train_idx = int(train_idx[0])
assumption_true = first_train_idx == 0

print("N_total:", N_total)
print("first_train_idx:", first_train_idx)
print("assumption (first train == original index 0):", assumption_true)

# %%
# 7.3 Resolve component run_ids used by selected metalearner via component_set_hash
groups_all = discover_ensembles(cfg.mlflow.experiment_name)
match = []
for ens_name_k, ids_k in groups_all.items():
    if component_set_hash(ids_k) == meta_component_set_hash:
        match.append((ens_name_k, ids_k))

if not match:
    raise RuntimeError(
        "Could not resolve component run_ids from component_set_hash. "
        "Ensure same experiment and grouping logic."
    )

ens_name_ref, run_ids_ref = match[0]
print("resolved ensemble:", ens_name_ref)
print("num component runs:", len(run_ids_ref))
if "linked" in globals() and isinstance(linked, pl.DataFrame) and linked.height > 0:
    linked_model_ids = sorted(linked["model_run_id"].to_list())
    print("same models as linked:", sorted(run_ids_ref) == linked_model_ids)

# %%
# 7.4 Load saved adapter_inputs artifacts (if available)
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
# 7.5 Manual reconstruction for first train sample of the metalearner
manual_idx = first_train_idx
labels_test = get_split_labels(cfg, "test")
y_manual = int(labels_test[manual_idx].item())

base_vecs_ref = []
profile_vecs_ref = []

for run_id_ref in run_ids_ref:
    inf_runs_ref = get_inference_run(cfg.mlflow.experiment_name, run_id_ref, "test")
    if inf_runs_ref.empty:
        raise RuntimeError(f"Missing test inference for {run_id_ref}")
    inf_run_id_ref = inf_runs_ref.iloc[0].run_id

    _, inf_tensors_ref = mh.load_inference_results(
        inf_run_id_ref, artifact_path="inference"
    )
    emb_ref = torch.from_numpy(inf_tensors_ref["embeddings"]).float().cpu()
    base_vecs_ref.append(emb_ref[manual_idx])

    prof_runs_ref = get_profile_run(
        cfg.mlflow.experiment_name, run_id_ref, meta_similarity_metric, "test"
    )
    if prof_runs_ref.empty:
        raise RuntimeError(
            f"Missing profile ({meta_similarity_metric}) for {run_id_ref}"
        )
    prof_run_id_ref = prof_runs_ref.iloc[0].run_id

    _, prof_tensor_ref = mh.load_profile_results(
        run_id=prof_run_id_ref,
        split="test",
        similarity_metric=meta_similarity_metric,
        artifact_path="profiles",
    )
    if prof_tensor_ref is None:
        raise RuntimeError(f"Missing profile tensor for {prof_run_id_ref}")
    profile_vecs_ref.append(prof_tensor_ref[manual_idx].float().cpu())

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
print("manual_idx (first train original index):", manual_idx)
print("manual_first_train_row shape:", manual_first_train_row.shape)

# %%
# 7.6 Compare manual reconstruction vs artifact row (when available)
print("assumption_true (first train == test index 0):", assumption_true)

if artifact_inputs_available:
    print(
        "artifact_first_train_original_idx == first_train_idx:",
        artifact_first_train_original_idx == first_train_idx,
    )
    if artifact_first_train_original_idx != first_train_idx:
        print("WARNING: artifact and recomputed split index differ")

    if manual_first_train_row.shape != artifact_first_train_row.shape:
        print(
            "shape mismatch:",
            manual_first_train_row.shape,
            artifact_first_train_row.shape,
        )
    else:
        abs_diff = np.abs(manual_first_train_row - artifact_first_train_row)
        print("max_abs_diff:", float(abs_diff.max()))
        print("mean_abs_diff:", float(abs_diff.mean()))
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
    print("No adapter_inputs artifact available for this metalearner run yet.")
    print("Re-run adapters after the logging patch, then re-run Section 7.")
