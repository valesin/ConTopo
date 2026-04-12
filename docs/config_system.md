# ConTopo Config System (Current Runtime)

This document describes the executable configuration model used by current ConTopo scripts.

Primary runtime sources:
- `conf/config.yaml`
- `conf/pipeline/default.yaml`
- `scripts/01_train_models.py` ... `scripts/05_train_adapters.py`
- `src/config/hash.py`
- `src/config/structured.py`

---

## 1) Runtime truth model

Configuration behavior is driven by Hydra composition of YAML groups.

- Entrypoint composition root: `conf/config.yaml`
- Pipeline orchestration: `main.py` reading `cfg.pipeline.steps`
- Stage scripts: each `scripts/*.py` consumes `cfg` directly

The practical runtime contract is therefore:

1. YAML group defaults + overrides compose into one `cfg`
2. Scripts read concrete keys from that composed `cfg`
3. Model idempotency is based on `cfg_hash` and step idempotency on `identity_hash`

`src/config/structured.py` is important for schema/testing and developer safety, but scripts do not call `register_configs()` directly at runtime.

---

## 2) Active Hydra group topology

From `conf/config.yaml`, active defaults are:

- `model`
- `loss`
- `dataset`
- `training`
- `runtime`
- `groups`
- `profiling`
- `analysis`
- `execution`
- `mlflow`
- `ensemble`
- `adapter`
- `pipeline`
- `_self_`

Top-level semantic keys:

- `schema_version`
- `trial`
- `seed` (`null` means auto-resolve as `100 + trial` in script logic)

---

## 3) What affects training identity vs what does not

`cfg_hash` in `src/config/hash.py` hashes the composed config after excluding `EXCLUDED_KEYS`.

### Included in `cfg_hash` (model identity)

- `schema_version`
- `trial`
- `seed` (resolved)
- `model.*`
- `loss.*`
- `dataset.*`
- `training.*`

### Excluded from `cfg_hash`

- `runtime`
- `mlflow`
- `storage`
- `hydra`
- `groups`
- `profiling`
- `analysis`
- `execution`
- `ensemble`
- `adapter`
- `pipeline`

This is validated in tests such as:
- `tests/test_cfg_hash.py`
- `tests/test_hydra_config.py`

---

## 4) Step-level idempotency fields

Step idempotency is not inferred from `cfg_hash`. It is explicitly enforced by `identity_hash(kind, **fields)` against `IDEMPOTENCY_REGISTRY` in `src/config/hash.py`.

Current required identity fields by kind:

- `model`: `schema_version`, `trial`, `seed`, plus wildcard groups `model.*`, `loss.*`, `dataset.*`, `training.*`
- `inference`: `trained_model_run_id`, `split`
- `category_similarity_profile`: `parent_run_id`, `anchor_spec_hash`, `similarity_metric`, `split`
- `diagnostics`: `parent_run_id`, `diagnostic_metric`, `split`
- `ensemble`: `component_set_hash`, `split`, `feature_type`, `method`
- `diversity`: `component_set_hash`, `diversity_metric`, `split`
- `consistency`: `component_set_hash`, `anchor_spec_hash`, `split`
- `metalearner`: `component_set_hash`, `split`, `feature_type`, `anchor_spec`, `meta_split_spec`, `similarity_metric`, `init_seed`, `profile_mask`, `meta_type`

`identity_hash` rejects:
- unknown fields
- missing required exact fields
- missing wildcard groups

---

## 5) Config + MLflow retrieval boundary

Run retrieval is repository-owned:

- `src/repositories/functional_run_repository.py`

Pipeline scripts call:

1. `setup_mlflow(cfg)`
2. `configure_run_repository(cfg.mlflow.tracking_uri, cfg.mlflow.experiment_name)`
3. `find_finished_model_run(...)` or `find_finished_identity_run(...)`

Do not add new run finder logic in `src/mlflow_utils.py`.

---

## 6) Structured config role (current state)

`src/config/structured.py` defines dataclass schemas and `register_configs()` for structured validation workflows, especially tests.

What it provides:
- group/type definitions for model, loss, dataset, training, runtime, groups, profiling, analysis, execution, mlflow, ensemble, adapter, pipeline
- `ConTopoConfig` top-level shape
- `register_configs()` to register with Hydra `ConfigStore`

Current runtime scripts are Hydra YAML-driven and do not require explicit `register_configs()` calls in each script file.

---

## 7) Safe parameter update workflow

When adding or changing a config parameter:

1. Update the relevant YAML group (`conf/<group>/*.yaml`)
2. Update script usage (read and apply the field)
3. Decide identity impact:
   - training semantics: ensure it falls under included `cfg_hash` groups
   - step semantics: add/update that step's `identity_hash` call and registry entry
4. Update telemetry contract if logged (`src/mlflow_schema_logger.py`)
5. Update tests (`tests/test_cfg_hash.py`, `tests/test_idempotency_registry.py`, or step-specific tests)

If parameter changes alter historical identity semantics, evaluate whether migration scripts are needed (`scripts/migrate_model_identity_hashes.py`, related migration utilities).

---

## 8) Runtime command patterns (current)

Use Hydra overrides (not legacy parser flags):

```bash
python main.py
python main.py +pipeline=small
python main.py pipeline.from_step=ensemble
python scripts/01_train_models.py loss.rho=0.05 trial=2
python scripts/02_cache_inference.py execution.split=val
```

---

## 9) Common mistakes to avoid

- Assuming `runtime` / `execution` / `groups` changes should create new model identity (they do not under `cfg_hash`)
- Changing step semantics without updating `IDEMPOTENCY_REGISTRY`
- Adding logged fields without updating telemetry schema
- Reintroducing retrieval wrappers outside the repository module

---

## 10) Related docs

- `ARCHITECTURE.md`
- `docs/telemetry_schema.md`
- `docs/idempotency.md`
- `CONTRIBUTING_AND_UPDATING.md`
