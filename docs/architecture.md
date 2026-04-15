# Architecture

## 1. Runtime model

ConTopo is a scripted research pipeline with a config-driven orchestrator.

- Orchestrator: `main.py`
- Step graph: `conf/pipeline/default.yaml` and `conf/pipeline/small.yaml`
- Stage scripts: `scripts/01_*.py` through `scripts/05_*.py`

Each stage is independently executable and tied together through MLflow run lineage and identity tags.

## 2. Data/control flow

Each stage consumes artifacts from MLflow (produced by an upstream stage) and
writes its own artifacts back to MLflow. Lineage is expressed through tags
(`parent_run_id`, `component_run_ids_csv`, etc.) rather than filesystem paths.

### 2.1 Stage inputs and outputs

| Stage | Run `kind` | Consumes (from where) | Produces |
|---|---|---|---|
| `01_train_models.py` | `model` | Dataset (torchvision/FFCV) | `e2e_best` model artifact, metrics (`test_accuracy`, `best_val_acc`, etc.) |
| `02_cache_inference.py` | `inference` | Model run (`e2e_best`) | `inference/{split}_inference_results.parquet`, `inference/{split}_tensors.npz` |
| `03_compute_profiles.py` | `category_similarity_profile` | Inference tensors (embeddings) | `profiles/{split}_{metric}_profiles.pt` |
| `03b_compute_diagnostics.py` | `diagnostics` | Model weights; inference tensors | `diagnostics/*.pt`, metrics (`morans_i`, `weight_norms_*`, etc.) |
| `04_run_ensemble.py` | `ensemble` | Inference parquets from component runs | `ensemble/composition_map.json`, `ensemble/*_inference.parquet` |
| `04b_compute_diversity.py` | `diversity` | Component inference parquets | Metrics only (`q_statistic`, `disagreement`, etc.) |
| `04c_compute_consistency.py` | `consistency` | Component inference tensors | `consistency/rsa_matrix.pt`, `consistency/run_id_ordering.json` |
| `05_train_adapters.py` | `metalearner` | Ensemble component artifacts; optional profiles | Adapter model, holdout parquets |

### 2.2 Stage summary

1. **Model training** (`kind=model`) — trains one configuration.
2. **Inference caching** (`kind=inference`) — materialises per-split tensors for downstream stages.
3. **Profiling + diagnostics** — `category_similarity_profile` consumes inference embeddings; `diagnostics` computes optional per-model metrics.
4. **Ensemble + analyses** — `ensemble` combines component logits; `diversity` and `consistency` compute group-level metrics.
5. **Meta-learner training** (`kind=metalearner`) — consumes ensemble component artifacts + optional profile-derived features.

## 3. Configuration system

Entry config: `conf/config.yaml`

Groups with distinct roles:

- **Model identity inputs**: `model`, `loss`, `dataset`, `training`, plus `schema_version`, `trial`, `seed`.
- **Execution/runtime controls**: `runtime`, `execution`, `mlflow`.
- **Post-training behavior**: `profiling`, `groups`, `ensemble`, `analysis`, `adapter`.
- **Orchestration layer**: `pipeline`, `sweeps`.

Important separation:

- Model identity hash (`cfg_hash`) excludes non-training groups (see `EXCLUDED_KEYS` in `src/config/hash.py`).

### 3.1 Dataset abstraction

The pipeline is dataset-agnostic. The active dataset is selected via the `dataset` config group:

```bash
python main.py dataset=imagenet100 model=resnet34_imagenet100 ...
```

The data loading layer (`src/data/loaders.py`) exposes two stable entry points:

- `get_dataset_loaders(cfg)` → `(train_loader, val_loader, test_loader)`
- `get_dataset_eval_loader(cfg, split, batch_size, num_workers)` → `DataLoader`

Both dispatch on `cfg.dataset.name` via `_DATASET_FACTORIES`, a registry of thin
torchvision-compatible dataset factory functions. Transform presets are selected by
`cfg.dataset.transforms.preset` from the registry in `src/data/transforms.py`.

Adding a new dataset requires:
1. One factory function + two registry entries in `src/data/loaders.py`.
2. A `conf/dataset/<name>.yaml` config file.
3. Optionally: a new transform preset in `src/data/transforms.py`.

See `contributing.md` §10 for the full checklist.

### 3.2 Data loading backend

`get_dataset_loaders` branches on `cfg.training.loading_backend`:

- **`torch`** (default) — standard `torch.utils.data.DataLoader`. Works for all
  datasets. All existing CIFAR-10 configs use this path.

- **`ffcv`** — FFCV binary `.beton` data loading for large-image datasets (e.g.
  ImageNet100). Requires the `ffcv` optional dependency group:

  ```bash
  uv sync --group ffcv
  ```

  The group installs `ffcv`, `antialiased-cnns` (for blurpool), and `cupy-cuda12x`
  (required by FFCV's GPU-side normalisation kernel).

  `.beton` files are generated **automatically on first use** via
  `src/data/beton_writer.py`. The path encodes dataset/split/resolution/quality
  parameters so the same file is safely reused across runs with the same config.

  When `training.progressive_res_min` and `training.progressive_res_max` are set,
  `get_dataset_loaders` returns a **list** of FFCV Loaders (one per discrete
  resolution step, low → high). The training script calls `_resolve_loader_for_epoch`
  each epoch to select the appropriate loader based on the ramp schedule.

**FFCV pipeline details:**

- Images are decoded directly onto GPU (via `ToDevice`) and normalised with a
  cupy-backed GPU kernel (`NormalizeImage`), producing `float16` tensors.
- The training and validation loops cast inputs to `float32` before the model
  forward pass (`.float()`) to match the model's weight dtype.
- FFCV's `IntDecoder` + `ToTensor` produces label tensors of shape `[B, 1]`;
  the training loops call `.squeeze()` to reduce them to `[B]`.
- The eval center-crop ratio is capped at `min(256/224, stored_size/image_size)`
  where `stored_size = min(training.beton.max_resolution, image_size)`. This prevents
  `CenterCropRGBImageDecoder` from requesting a region larger than the stored
  image (critical for small datasets like CIFAR-10 where stored size equals
  `image_size`).

**Hash placement:** `loading_backend` and the beton format settings
(`beton.max_resolution`, `beton.jpeg_quality`, `beton.compress_probability`) live
in `TrainingConfig` and are therefore **hash-included** — two runs that differ only
in `loading_backend` or beton format settings produce different model identity hashes.
The beton storage location (`runtime.beton.dir`) is hash-excluded (operational only).

Reference:
- `src/data/ffcv_pipelines.py` — FFCV augmentation pipeline builders
- `src/data/beton_writer.py` — on-demand beton generation
- `conf/sweeps/training_rho_imagenet100_ffcv.yaml` — full FFCV recipe sweep
- `docs/ffcv_param_assumptions.md` — migration guide for the new training params

### 3.3 Training config validation

`src/config/validation.py` runs as the first operation inside
`scripts/01_train_models.py`. It enforces the **conditional fields** principle:
some `TrainingConfig` fields are only meaningful when a parent feature is
active (e.g. `lr_peak_epoch` requires `scheduler=cyclic`). Violations raise
`ValueError` at startup, before any work is done.

Full rule table and rationale → [`config_system.md`](config_system.md) §5.

## 4. MLflow architecture boundaries

### 4.1 Retrieval boundary (SSOT)

All run lookup logic must go through:

- `src/repositories/functional_run_repository.py`

Core API:

- `configure_run_repository(...)`
- `search_runs(...)`
- `find_finished_identity_run(...)`
- `find_finished_model_run(...)`
- `get_run(...)`

### 4.2 Logging/utilities boundary

`src/mlflow_utils.py` provides setup/logging/artifact helpers, not run retrieval ownership.

The `log_resolved_config` helper logs the fully-resolved Hydra config as
`config/resolved_config.yaml` (stable artifact path). This artifact is used by
`scripts/migrations/rehash_identities.py` to reconstruct identity hashes after
schema changes.

### 4.3 Telemetry contract boundary

`src/mlflow_schema_logger.py` defines required telemetry per run `kind` and
validates run completeness on successful exit. Full schema contract →
[`telemetry_schema.md`](telemetry_schema.md).

## 5. Idempotency model

Every stage computes an `identity_hash` over its semantic inputs and skips if a
matching `FINISHED` MLflow run already exists (unless `execution.force=true`).
The registry that defines each run kind's identity fields lives at
`IDEMPOTENCY_REGISTRY` in `src/config/hash.py`.

Full idempotency model (registry semantics, step-by-step behavior, migration
protocol) → [`idempotency.md`](idempotency.md).

## 6. Ensemble discovery architecture

Discovery is config-driven and dynamic:

- `src/ensemble/selector.py`
- group controls in `conf/groups/default.yaml`

Grouping is based on finished model params and supports optional k-combination expansion via `groups.sample_size`.

## 7. Notebook analysis layer

Analysis helper module:

- `notebooks/mlflow/mlflow_helpers.py`

This layer is allowed to consume `src/*`, but pipeline scripts should not depend on notebook-specific helper modules.

## 8. Invariants to preserve

1. **Repository-first retrieval**: no duplicate MLflow finder wrappers outside repository module.
2. **Schema-aligned logging**: any new logged param/tag/metric/artifact must appear in `TELEMETRY_SCHEMA` in `src/mlflow_schema_logger.py`. Add it to `"optional"` so existing runs still pass validation.
3. **Identity parity**: if a parameter affects semantic outputs (lives in a hash-included group), existing run identity hashes must be migrated before deployment. See `contributing.md` §11.
4. **Config truth in YAML**: operational behavior is defined by active Hydra YAML groups + script usage.
5. **Conditional fields**: fields that are only meaningful when a parent feature is active must default to `None` and be enforced by `src/config/validation.py` — never backfill a fictitious value.
