# Configuration System

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

## 8) Adding a new parameter — three cases

Before adding a field, decide which of the three cases it falls into. Each has
a different cost and migration path. The full migration protocol lives in
[`contributing.md`](contributing.md) §11; this section is the canonical decision
guide.

### Case A — Hash-included parameter (mandatory migration)

**Criteria:** The param changes what the trained model is (training recipe,
loss, model, dataset) or what a downstream run identifies. It lives in a group
covered by the wildcard tuple `("model.*", "loss.*", "dataset.*", "training.*")`
inside `model_identity_fields()` in `src/config/hash.py`, or by
`IDEMPOTENCY_REGISTRY` for that run kind.

**Cost:** Every existing FINISHED run's `identity_hash` becomes stale.
Training will no longer recognise existing runs and will restart them unless
you migrate.

**Checklist:**
1. Add field to the dataclass in `src/config/structured.py` with the current default.
2. Add to `conf/<group>/default.yaml` with a brief comment.
3. Wire up behavior in the script(s) that consume it.
4. Log the param (via `log_params` from `src/mlflow_schema_logger`; scripts
   import it under the alias `schema_log_params`) and add it to `"optional"` in
   `TELEMETRY_SCHEMA` in `src/mlflow_schema_logger.py`.
5. Write `scripts/migrations/specs/<feature>.yaml` with the migration default.
6. Run `scripts/migrations/backfill_params.py --spec ... --apply`.
7. Run `scripts/migrations/rehash_identities.py --apply`.
8. Verify: re-run an existing config → "Run already exists, skipping."
9. Update tests in `tests/test_cfg_hash.py`.

### Case B — Hash-excluded parameter (optional observability migration)

**Criteria:** The param is purely operational — storage location, workers,
verbosity, MLflow routing, etc. Lives in `runtime`, `execution`, `mlflow`,
`groups`, etc. (any key in `EXCLUDED_KEYS`).

**Cost:** Zero identity cost. Existing runs keep their hashes.

**Checklist:** Same as Case A steps 1–4. Step 5 (backfill spec) is optional:
run it only if you want existing MLflow runs to carry the param for query
convenience. No rehash needed.

### Case C — Conditional parameter (A or B + validation rule)

**Criteria:** Only meaningful when a parent feature is active (e.g.
`lr_peak_epoch` requires `scheduler=cyclic`). Applies to both hash-included
and hash-excluded fields.

**Extra requirement:** Default must be `None` (not a fictitious numeric value).
Add a rule in `src/config/validation.py` enforcing "set iff parent is active".
Fail early at startup.

**Why `None`?** Backfilling a fictitious value (e.g. `lr_peak_epoch=2` for
runs that never used the cyclic scheduler) misrepresents what actually ran.
`None` means "feature was off".

**Current examples:**
- `lr_peak_epoch` (conditional on `scheduler=cyclic`)
- `progressive_res_start_ramp/end_ramp` (conditional on `progressive_res_min`)
- `beton.max_resolution/jpeg_quality/compress_probability` (conditional on `loading_backend=ffcv`)

See [`ffcv_param_assumptions.md`](ffcv_param_assumptions.md) for a full worked
example of a Case A + C migration.

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

## 11) MLflow param name stripping and filter keys

### How params are logged

`log_params` (from `src/mlflow_schema_logger`; aliased as `schema_log_params`
in scripts) logs each param using the **field name only** — the Hydra
config-group prefix is stripped. Examples:

| Hydra path | MLflow param key |
|---|---|
| `training.epochs` | `epochs` |
| `training.loading_backend` | `loading_backend` |
| `loss.rho` | `rho` |
| `loss.topology` | `topology` |
| `dataset.name` | (see `TELEMETRY_SCHEMA` for exact key) |

Tag fields (e.g. `trial`, `kind`, `cfg_hash`) are logged via `set_tags` and
are accessed in MLflow as `tags.<name>`.

### Consequence for `groups.filter`

Keys in `conf/groups/*.yaml` `filter` dicts are appended **verbatim** to the
MLflow filter string. They must use full MLflow entity paths:

```yaml
# Correct — full entity prefix
filter: {"params.epochs": "1", "tags.trial": "3"}

# Wrong — Hydra path, not an MLflow entity path
filter: {"training.epochs": "1"}
```

Values must be strings; MLflow stores all params and tags as strings.

### Global uniqueness requirement

Because the config-group prefix is stripped on logging, two config groups
cannot define a field with the same leaf name. For example, adding
`training.rho` alongside the existing `loss.rho` would cause both to log
as `rho`, silently overwriting one another.

When adding a new parameter, verify its field name is globally unique across
all config groups before proceeding with the migration checklist in §8.

---

## 12) Related docs

- `architecture.md`
- `docs/telemetry_schema.md`
- `docs/idempotency.md`
- `contributing.md`
