# Migration System

This document is the single reference for **how ConTopo migrations work** when
config schema changes are introduced.

It complements (does not replace):
- `docs/config_system.md` (how to classify a new parameter)
- `docs/idempotency.md` (identity-hash semantics)
- `docs/contributing.md` (safe change procedure)

---

## 1) Why migrations exist

Model idempotency is defined by `identity_hash("model", ...)`, which includes:
- `model.*`
- `loss.*`
- `dataset.*`
- `training.*`
- plus `schema_version`, `trial`, `seed`

If you add a new field under one of those hash-included groups, every historical
model run has an out-of-date identity hash unless migrated. Without migration,
`01_train_models.py` misses existing FINISHED runs and retrains duplicates.

---

## 2) Migration building blocks

### 2.1 Spec-driven param backfill

Script: `scripts/migrations/backfill_params.py`

Purpose:
- read a YAML spec under `scripts/migrations/specs/*.yaml`
- find FINISHED model runs in an experiment
- add missing params with migration-default values
- never overwrite already-set params (idempotent)

Important behavior:
- defaults are stored as strings (MLflow param format)
- dry-run is default
- writes happen only with `--apply`
- requires explicit `--experiment`
- `--tracking-uri` defaults to `sqlite:///outputs/mlflow.db`

### 2.2 Identity-hash rehash

Script: `scripts/migrations/rehash_identities.py`

Purpose:
- recompute `tags.identity_hash` for FINISHED model runs after hash-included
  schema changes

How it works per run:
1. downloads `config/resolved_config.yaml` artifact (with fallback discovery)
2. canonicalizes sections against current structured config classes
3. recomputes model identity hash from canonical fields
4. updates `identity_hash` tag only when changed (`--apply` required)

Important behavior:
- dry-run is default
- writes happen only with `--apply`
- requires explicit `--experiment`
- `--tracking-uri` defaults to `sqlite:///outputs/mlflow.db`

---

## 3) When to run which migration

## 3.1 Hash-included change (`training.*`, `model.*`, `loss.*`, `dataset.*`)

Run both scripts, in this order:
1. param backfill (`backfill_params.py`)
2. identity rehash (`rehash_identities.py`)

This is mandatory for correctness.

## 3.2 Hash-excluded change (`runtime.*`, `execution.*`, etc.)

Only param backfill is relevant, and optional (recommended for observability).
No identity rehash is needed.

## 3.3 Conditional fields

For fields that are only meaningful under a parent switch (example:
`topography_type=topoloss`), migration defaults should reflect historical truth:
- usually `"None"` when the feature was previously inactive
- never use fictitious numeric defaults for inactive historical runs

Runtime validation must enforce:
- required when parent feature is active
- forbidden (orphaned) when parent feature is inactive

Validation lives in `src/config/validation.py`.

---

## 4) Spec format and examples

Migration spec format:

```yaml
description: "Short migration description"
params:
  param_name: "string_value"
  another_param: "None"
```

Current repo examples:
- `scripts/migrations/specs/ffcv_training_params.yaml`
- `scripts/migrations/specs/topoloss_loss_params.yaml`

---

## 5) Standard operator workflow

1. Write assumptions doc first
   - `docs/<feature>_param_assumptions.md`
   - document old hardcoded behavior and migration defaults

2. Add/update migration spec
   - `scripts/migrations/specs/<feature>.yaml`

3. Dry-run backfill

```bash
uv run scripts/migrations/backfill_params.py \
  --spec scripts/migrations/specs/<feature>.yaml \
  --experiment <experiment_name> \
  --tracking-uri <tracking_uri>
```

4. Apply backfill

```bash
uv run scripts/migrations/backfill_params.py \
  --spec scripts/migrations/specs/<feature>.yaml \
  --experiment <experiment_name> \
  --tracking-uri <tracking_uri> \
  --apply
```

5. If hash-included change: dry-run rehash

```bash
uv run scripts/migrations/rehash_identities.py \
  --experiment <experiment_name> \
  --tracking-uri <tracking_uri>
```

6. If hash-included change: apply rehash

```bash
uv run scripts/migrations/rehash_identities.py \
  --experiment <experiment_name> \
  --tracking-uri <tracking_uri> \
  --apply
```

7. Verify idempotency
   - re-run representative training config
   - confirm script reports existing FINISHED run and skips retraining

---

## 6) Local vs remote tracking

If your MLflow server is remote, pass `--tracking-uri` explicitly to both
migration scripts. They do not auto-read your Hydra config.

Examples:

```bash
# Local default SQLite
--tracking-uri sqlite:///outputs/mlflow.db

# Remote MLflow server
--tracking-uri http://<host>:5000
```

Always run migrations against the same experiment and tracking backend used by
the original runs.

---

## 7) Failure modes and recovery

### 7.1 Duplicate retraining after schema change

Symptoms:
- training starts for configs that were previously completed
- nearly identical runs with different `identity_hash`

Fix:
1. stop duplicate jobs
2. run missing migration(s): backfill + (if needed) rehash
3. relaunch training and confirm idempotency hits
4. clean up duplicate runs per project policy

### 7.2 Backfill appears to do nothing

Checklist:
- correct `--experiment`?
- correct `--tracking-uri`?
- runs are FINISHED and `tags.kind = model`?
- params already exist (script should print SKIP)?

### 7.3 Rehash cannot compute run hash

Common cause:
- missing config artifact under `config/`

Script behavior:
- logs run-level error and continues

Action:
- inspect the affected run artifact set and repair manually if needed

---

## 8) Testing expectations for migration-related PRs

For migration changes, include targeted checks for:
- config validation behavior (`src/config/validation.py`)
- idempotency registry/hash behavior when relevant
- migration dry-run output sanity on a representative experiment

Use smoke scope for execution validation (`training.epochs=1`).

---

## 9) Current migration assets in this repo

Scripts:
- `scripts/migrations/backfill_params.py`
- `scripts/migrations/rehash_identities.py`

Specs:
- `scripts/migrations/specs/ffcv_training_params.yaml`
- `scripts/migrations/specs/topoloss_loss_params.yaml`

Worked assumptions docs:
- `docs/ffcv_param_assumptions.md`
- `docs/topoloss_param_assumptions.md`

---

## 10) Quick decision table

| Change type | Backfill params | Rehash identity_hash |
|---|---:|---:|
| New field in `training.*` / `model.*` / `loss.*` / `dataset.*` | Required | Required |
| New field in `runtime.*` / `execution.*` | Optional (recommended) | Not needed |
| Conditional field under active parent feature | Required when hash-included | Required when hash-included |
| Conditional field under hash-excluded parent feature | Optional (recommended) | Not needed |

When in doubt, treat the change as hash-included until proven otherwise and run
a dry-run migration first.
