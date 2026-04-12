# Architecture

## 1. Runtime model

ConTopo is a scripted research pipeline with a config-driven orchestrator.

- Orchestrator: `main.py`
- Step graph: `conf/pipeline/default.yaml` and `conf/pipeline/small.yaml`
- Stage scripts: `scripts/01_*.py` through `scripts/05_*.py`

Each stage is independently executable and tied together through MLflow run lineage and identity tags.

## 2. Data/control flow

1. **Model training** (`kind=model`)
   - produces best model artifact (`e2e_best`) and core training/test metrics.
2. **Inference caching** (`kind=inference`)
   - consumes model run, logs `{split}_inference_results.parquet` and `{split}_tensors.npz`.
3. **Profiling + diagnostics**
   - `kind=category_similarity_profile` consumes inference embeddings.
   - `kind=diagnostics` computes optional per-model diagnostics.
4. **Ensemble + analyses**
   - `kind=ensemble` combines component logits.
   - `kind=diversity` and `kind=consistency` compute group-level analyses.
5. **Meta-learner training** (`kind=metalearner`)
   - consumes ensemble component artifacts and optional profile-derived features.

## 3. Configuration system

Entry config: `conf/config.yaml`

Groups with distinct roles:

- **Model identity inputs**: `model`, `loss`, `dataset`, `training`, plus `schema_version`, `trial`, `seed`.
- **Execution/runtime controls**: `runtime`, `execution`, `mlflow`.
- **Post-training behavior**: `profiling`, `groups`, `ensemble`, `analysis`, `adapter`.
- **Orchestration layer**: `pipeline`, `sweeps`.

Important separation:

- Model identity hash (`cfg_hash`) excludes non-training groups (see `EXCLUDED_KEYS` in `src/config/hash.py`).

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

### 4.3 Telemetry contract boundary

`src/mlflow_schema_logger.py` defines required telemetry per run `kind` and validates run completeness on successful exit.

## 5. Idempotency model

Identity source of truth:

- `IDEMPOTENCY_REGISTRY` in `src/config/hash.py`

`identity_hash(kind, **fields)` enforces:

- no unknown fields,
- all required exact fields present,
- all wildcard groups represented where configured.

Pipeline scripts compute an identity hash for their semantic inputs, then skip if a matching `FINISHED` run exists (unless `execution.force=true`).

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
2. **Schema-aligned logging**: any new param/tag/metric/artifact must be reflected in telemetry schema.
3. **Identity parity**: if a parameter changes semantic outputs, identity fields and tests must be updated.
4. **Config truth in YAML**: operational behavior is defined by active Hydra YAML groups + script usage.
