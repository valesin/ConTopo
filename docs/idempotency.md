# ConTopo Idempotency (Current State)

This document describes how run uniqueness is defined and enforced in the current pipeline.

Primary implementation sources:
- `src/config/hash.py`
- `src/repositories/functional_run_repository.py`
- `scripts/01_train_models.py` ... `scripts/05_train_adapters.py`
- `tests/test_idempotency_registry.py`
- `tests/test_cfg_hash.py`

---

## 1) Core concepts

### 1.1 Model config hash: `cfg_hash(cfg)`

Implemented in `src/config/hash.py`.

Purpose:
- deterministic hash of experiment-semantic training config
- used in model run tagging and model-run lookup flow

Important behavior:
- hashes composed Hydra config with excluded top-level keys removed (`EXCLUDED_KEYS`)
- order-invariant canonicalization
- returns 16-char SHA-256 prefix

### 1.2 Step identity hash: `identity_hash(kind, **fields)`

Implemented in `src/config/hash.py`.

Purpose:
- deterministic hash for each pipeline step's semantic inputs
- strict field validation via `IDEMPOTENCY_REGISTRY`

Important behavior:
- unknown field -> `ValueError`
- missing required exact field -> `ValueError`
- missing wildcard group coverage (for `model.*` etc.) -> `ValueError`

### 1.3 Repository lookup contract

Implemented in `src/repositories/functional_run_repository.py`.

Pipeline scripts use:
- `configure_run_repository(...)`
- `find_finished_model_run(...)`
- `find_finished_identity_run(kind, identity_hash)`

Idempotency check pattern is always against `FINISHED` runs in configured experiment context.

---

## 2) Registry-defined identity fields

`IDEMPOTENCY_REGISTRY` is the source of truth.

Current identity fields:

- `model`
  - `schema_version`, `trial`, `seed`, `model.*`, `loss.*`, `dataset.*`, `training.*`
- `inference`
  - `trained_model_run_id`, `split`
- `category_similarity_profile`
  - `parent_run_id`, `anchor_spec_hash`, `similarity_metric`, `split`
- `diagnostics`
  - `parent_run_id`, `diagnostic_metric`, `split`
- `ensemble`
  - `component_set_hash`, `split`, `feature_type`, `method`
- `diversity`
  - `component_set_hash`, `diversity_metric`, `split`
- `consistency`
  - `component_set_hash`, `anchor_spec_hash`, `split`
- `metalearner`
  - `component_set_hash`, `split`, `feature_type`, `anchor_spec`, `meta_split_spec`, `similarity_metric`, `init_seed`, `profile_mask`, `meta_type`

The registry and telemetry kinds are tested for parity in `tests/test_idempotency_registry.py`.

---

## 3) Step-by-step idempotency behavior in scripts

## 3.1 Step 01 — train (`scripts/01_train_models.py`)

- Computes `cfg_hash(cfg)` for tagging
- Computes model identity through repository helper (`find_finished_model_run(cfg, seed)` uses `model_identity_fields` + `identity_hash("model", ...)`)
- If matching `FINISHED` model run exists, training is skipped

## 3.2 Step 02 — inference (`scripts/02_cache_inference.py`)

Identity fields:
- `trained_model_run_id`
- `split`

Flow:
- locate parent model run
- compute inference identity hash
- skip if `find_finished_identity_run("inference", hash)` finds a finished run (unless `execution.force=true`)

## 3.3 Step 03 — profiles (`scripts/03_compute_profiles.py`)

Identity fields:
- `parent_run_id`
- `anchor_spec_hash`
- `similarity_metric`
- `split`

One run per metric and anchor spec for a model/split combination.

## 3.4 Step 03b — diagnostics (`scripts/03b_compute_diagnostics.py`)

Identity fields:
- `parent_run_id`
- `diagnostic_metric`
- `split`

One run per diagnostic metric. Split is part of identity (important current behavior).

## 3.5 Step 04 — ensemble (`scripts/04_run_ensemble.py`)

Identity fields:
- `component_set_hash`
- `split`
- `feature_type`
- `method`

Notes:
- `component_set_hash` is deterministic hash of sorted component run IDs
- `method` is part of identity (prevents cross-method collisions)

## 3.6 Step 04b — diversity (`scripts/04b_compute_diversity.py`)

Identity fields:
- `component_set_hash`
- `diversity_metric`
- `split`

One run per metric per component set and split.

## 3.7 Step 04c — consistency (`scripts/04c_compute_consistency.py`)

Identity fields:
- `component_set_hash`
- `anchor_spec_hash`
- `split`

One run per component set/anchor spec/split.

## 3.8 Step 05 — metalearner (`scripts/05_train_adapters.py`)

Identity fields:
- `component_set_hash`
- `split`
- `feature_type`
- `anchor_spec`
- `meta_split_spec`
- `similarity_metric`
- `init_seed`
- `profile_mask`
- `meta_type`

Notes:
- `meta_type` is currently part of identity (prevents LR-vs-MLP collisions)
- `meta_split_spec` is serialized from Hydra config for deterministic split semantics

---

## 4) What is and is not part of model identity

Model identity (`cfg_hash` + model identity fields) includes:
- model/loss/dataset/training semantics
- `schema_version`, `trial`, `seed`

Model identity excludes:
- runtime infra knobs (`runtime`, `mlflow`, etc.)
- post-training analysis config (`groups`, `profiling`, `analysis`, `ensemble`, `adapter`, `pipeline`, `execution`)

See `EXCLUDED_KEYS` in `src/config/hash.py` and tests in `tests/test_cfg_hash.py`.

---

## 5) How to change idempotency safely

For the full decision framework when **adding new parameters** (covering hash-included,
hash-excluded, and conditional cases), see `CONTRIBUTING_AND_UPDATING.md` §11.

When **changing step semantics** (modifying what an existing identity hash covers):

1. Update `IDEMPOTENCY_REGISTRY` in `src/config/hash.py`
2. Update all `identity_hash(...)` call sites in affected scripts/modules
3. Ensure MLflow tags/params still provide traceable context
4. Update tests:
   - `tests/test_idempotency_registry.py`
   - relevant step tests
5. If model-level semantics changed, validate `cfg_hash` behavior via `tests/test_cfg_hash.py`

When telemetry keys are added/removed, also update:
- `src/mlflow_schema_logger.py`
- `docs/telemetry_schema.md`

### 5.1 Schema migration protocol

When a new parameter is introduced, whether migration is **required** or only
**recommended** depends on where the parameter lives:

| Parameter location | Idempotency impact | Identity hash migration |
|---|---|---|
| Hash-included group (`training.*`, `model.*`, `loss.*`, `dataset.*`) | **Breaks** — all existing run hashes are invalid | **Mandatory** |
| Hash-excluded group (`runtime.*`, `execution.*`, etc.) | None — existing hashes remain valid | **Optional** (for observability) |

In both cases, running the param backfill script is recommended so that historical
runs have a visible, queryable record of what value was hardcoded before the param
existed.

#### For hash-included params (mandatory migration)

The model `identity_hash` covers `training.*` with a wildcard. **Every new field
added to `TrainingConfig` invalidates all existing model run hashes.** Before
merging or deploying any such schema change:

1. **Write a param assumptions document** (`docs/<feature>_param_assumptions.md`)
   recording each new field, the behaviour it was previously hardcoded to, and the
   migration default that preserves the old behaviour. Write this *before* any code
   change.

2. **Write a migration spec** (`scripts/migrations/specs/<feature>.yaml`) that
   lists each new param and its migration default. Run the generic backfill script
   against it — dry-run by default; `--apply` to write:

   ```bash
   uv run scripts/migrations/backfill_params.py \
       --spec scripts/migrations/specs/<feature>.yaml \
       --experiment <experiment_name> [--apply]
   ```

   See `scripts/migrations/specs/ffcv_training_params.yaml` as the canonical example.

3. **Run the identity hash rehash script**
   (`scripts/migrations/rehash_identities.py`) that recomputes `tags.identity_hash`
   for every FINISHED model run by downloading its stored `config/resolved_config.yaml`
   artifact and merging it with the current structured config defaults via
   `_canonical_section()`. New fields automatically receive their migration defaults.

   **Artifact path note:** `src/mlflow_utils.log_resolved_config` logs the artifact
   at the stable path `config/resolved_config.yaml`. The rehash script tries this
   path first and falls back to listing `config/` and taking the first `.yaml` found,
   for compatibility with runs logged before this naming was fixed.

4. **Run both scripts** against all affected experiments before deploying the new
   code. Param backfill first, then identity hash rehash.

5. Verify idempotency is restored: re-running `01_train_models.py` with an existing
   config should find the idempotency hit and skip training.

#### For hash-excluded params (optional observability migration)

No identity hash rehash is needed. Only the param backfill script is relevant, and
only if you want the old hardcoded value to appear in MLflow for historical runs.

1. Write a param backfill script (or extend the existing one) following the same
   template. Document the old hardcoded behaviour in the assumptions doc under a
   "Runtime (hash-excluded)" section.

2. Run the backfill (no rehash step).

See `docs/ffcv_param_assumptions.md` §"New Params: Runtime" as an example.

---

## 6) Verification checklist

Before merging idempotency changes:

- [ ] Registry updated (`IDEMPOTENCY_REGISTRY`)
- [ ] Script call sites updated
- [ ] No legacy retrieval wrappers introduced
- [ ] Telemetry schema aligned
- [ ] Tests updated/passing (`test_idempotency_registry`, `test_cfg_hash`)
- [ ] One smoke rerun confirms expected skip/recompute behavior

---

## 7) Related docs

- `docs/config_system.md`
- `docs/telemetry_schema.md`
- `ARCHITECTURE.md`
- `CONTRIBUTING_AND_UPDATING.md`
