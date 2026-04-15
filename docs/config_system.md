# ConTopo Config System (Current Runtime)

This document describes the executable configuration model used by current ConTopo scripts.

Primary runtime sources:
- `conf/config.yaml`
- `conf/pipeline/default.yaml`
- `scripts/01_train_models.py` ... `scripts/05_train_adapters.py`
- `src/config/hash.py`
- `src/config/structured.py`
- `src/config/validation.py`

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

- `runtime` (only `beton.dir` remains here after the hash coherence fix; all
  format-affecting fields moved to `training.*`)
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

**Hash coherence note:** `loading_backend` and the beton format fields
(`beton.max_resolution`, `beton.jpeg_quality`, `beton.compress_probability`) live
in `TrainingConfig`, not `RuntimeConfig`, because they directly affect the training
data pipeline. Two runs that differ in `loading_backend` or beton format produce
different models and must have distinct identity hashes. The beton storage directory
(`runtime.beton.dir`) is the only remaining beton-related field in `RuntimeConfig`
because it is purely operational (does not affect what is stored, only where).

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

## 5) Training config validation (fail-early)

`src/config/validation.py` defines `validate_training_config(cfg)`, called as the
**first operation** in `scripts/01_train_models.py` before any data loading or model
building. It enforces consistency across the composed training config.

### Conditional fields principle

Some `TrainingConfig` fields are only meaningful when a parent feature is active.
These are typed as `Optional` and default to `None` rather than a fictitious value:

| Field | Active when |
|---|---|
| `training.lr_peak_epoch` | `training.scheduler = cyclic` |
| `training.progressive_res_start_ramp` | `training.progressive_res_min` is set |
| `training.progressive_res_end_ramp` | `training.progressive_res_min` is set |
| `training.beton.max_resolution` | `training.loading_backend = ffcv` |
| `training.beton.jpeg_quality` | `training.loading_backend = ffcv` |
| `training.beton.compress_probability` | `training.loading_backend = ffcv` |

The validator rejects:
- a conditional field set when its parent feature is inactive (orphaned field)
- a parent feature active but its required conditional field is `None`
- `lr_tta=True` or progressive resolution without `loading_backend=ffcv`
- `progressive_res_min >= progressive_res_max`
- `loading_backend=ffcv` without all three beton format fields set

This ensures config state accurately represents what ran, which matters for identity
hash integrity and migration correctness.

---

## 6) Config + MLflow retrieval boundary

Run retrieval is repository-owned:

- `src/repositories/functional_run_repository.py`

Pipeline scripts call:

1. `setup_mlflow(cfg)`
2. `configure_run_repository(cfg.mlflow.tracking_uri, cfg.mlflow.experiment_name)`
3. `find_finished_model_run(...)` or `find_finished_identity_run(...)`

Do not add new run finder logic in `src/mlflow_utils.py`.

---

## 7) Structured config role (current state)

`src/config/structured.py` defines dataclass schemas and `register_configs()` for structured validation workflows, especially tests.

What it provides:
- group/type definitions for model, loss, dataset, training, runtime, groups, profiling, analysis, execution, mlflow, ensemble, adapter, pipeline
- `ConTopoConfig` top-level shape
- `register_configs()` to register with Hydra `ConfigStore`
- `TrainingBetonConfig` nested under `TrainingConfig` — FFCV beton **format** settings
  (`max_resolution`, `jpeg_quality`, `compress_probability`); hash-included
- `BetonConfig` nested under `RuntimeConfig` — beton **storage** location (`dir` only);
  hash-excluded

Current runtime scripts are Hydra YAML-driven and do not require explicit `register_configs()` calls in each script file.

---

## 8) Safe parameter update workflow

The full protocol for adding a new parameter depends on whether it is hash-included,
hash-excluded, and/or conditional. The authoritative guide is
`CONTRIBUTING_AND_UPDATING.md` §11. This section is a quick reference.

### Always required

1. Update the relevant YAML group (`conf/<group>/*.yaml`)
2. Update `src/config/structured.py` to match
3. Update script usage (read and apply the field)
4. If the param is logged to MLflow: add it to the `"optional"` list in
   `TELEMETRY_SCHEMA` in `src/mlflow_schema_logger.py`, and log it via
   `schema_log_params` in the script

### If hash-included (param lives in `training.*`, `model.*`, `loss.*`, `dataset.*`)

The param changes `identity_hash` for all existing runs. **Migration is mandatory.**

5. Write `docs/<feature>_param_assumptions.md` documenting the migration default
6. Write `scripts/migrations/specs/<feature>.yaml`; run `scripts/migrations/backfill_params.py --spec ... --apply` to backfill MLflow params
7. Run `scripts/migrations/rehash_identities.py --apply` to rehash identity tags
8. Verify idempotency: re-run an existing config → "already FINISHED, skipping"
9. Update tests in `tests/test_cfg_hash.py`

### If hash-excluded (param lives in `runtime.*`, `execution.*`, etc.)

The param does not break idempotency. No identity hash migration needed.

5. *(Optional)* Write a param backfill script for observability — lets you query
   MLflow for the old hardcoded value on historical runs

### If conditional (only valid when a parent feature is active)

6. Type as `Optional[T] = None` in the struct; use `null` in YAML
7. Add validation rules to `src/config/validation.py` (reject orphaned + missing)
8. Migration default must be `"None"`, not a fictitious numeric value

---

## 9) Runtime command patterns (current)

Use Hydra overrides (not legacy parser flags):

```bash
python main.py
python main.py pipeline=small
python main.py pipeline.from_step=ensemble
python scripts/01_train_models.py loss.rho=0.05 trial=2
python scripts/02_cache_inference.py execution.split=val
```

---

## 10) Common mistakes to avoid

- Assuming `runtime` / `execution` / `groups` changes should create new model identity (they do not under `cfg_hash`)
- Changing step semantics without updating `IDEMPOTENCY_REGISTRY`
- Adding logged fields without updating telemetry schema
- Reintroducing retrieval wrappers outside the repository module
- Setting a fictitious default value for a conditional field (e.g. `lr_peak_epoch=2` on a run that used `scheduler=none`) — use `None` and enforce via `validate_training_config`

---

## 11) Related docs

- `ARCHITECTURE.md`
- `docs/telemetry_schema.md`
- `docs/idempotency.md`
- `CONTRIBUTING_AND_UPDATING.md`
