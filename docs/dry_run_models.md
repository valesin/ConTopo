# Dry Run — Model Training Idempotency Inspector

`scripts/01_dry_run_models.py` is a zero-side-effect inspection tool for step 01.
It performs the exact same config resolution as `01_train_models.py` — same Hydra
decorator, same `validate_training_config`, `resolve_seed`, `cfg_hash`,
`find_finished_model_run` — so the identity hash it computes is guaranteed to be
the same hash the real script would produce.

**No training happens. No MLflow runs are created or modified.**

Primary sources:
- `scripts/01_dry_run_models.py`
- `scripts/01_train_models.py` (reference — mirrored setup sequence)
- `src/config/hash.py` (`model_identity_fields`, `cfg_hash`)
- `src/repositories/functional_run_repository.py`

---

## When to use it

| Situation | What to look for |
|---|---|
| A config that should have been skipped is training again | STALE IDENTITY HASH banner → run the rehash migration |
| A new config is unexpectedly triggering a training run | Diff table → look for `*`-marked params that differ |
| You want to preview what a config resolves to before committing | WOULD SKIP output confirms the hash hits |
| You changed a schema/default and want to verify nothing broke | Check all affected trials → all should WOULD SKIP |

---

## How it works

### 1 — Config resolution (same as `01_train_models.py`)

```
validate_training_config(cfg)
  ↓
resolve_seed(cfg)          # seed = cfg.seed or 100 + trial
  ↓
cfg_hash(cfg)              # SHA-256 over model/loss/dataset/training.*
  ↓
setup_mlflow(cfg)
configure_run_repository(...)
  ↓
find_finished_model_run(cfg, seed)   # queries tags.identity_hash in MLflow
```

If a FINISHED run is found → **WOULD SKIP**.
If not → **WOULD TRAIN**, then candidate search begins.

### 2 — Candidate search

All FINISHED model runs in the experiment are fetched (optionally filtered via
`dry_filter.*`). Each is scored by how many of its logged params match the
proposed config. Runs missing more than half the expected params are excluded
(schema-mismatched old runs).

All candidate runs are printed, sorted by score descending.

### 3 — Diff output

For each candidate, only params that differ are shown.
Params marked with `*` are part of the identity hash — a difference in any of
them explains why the run was not matched.

### 4 — Stale hash detection

If a candidate matches **all** logged params but its stored `tags.identity_hash`
differs from the computed one, the script prints a `!!!` banner:

```
!!! run 827d7b4d834a (33/33 params match) — STALE IDENTITY HASH !!!
  All logged params match but identity_hash differs.
  Proposed : b18b28caf2c590f3
  Stored   : 780e5573db0a1249
  Fix      : uv run scripts/migrations/rehash_identities.py --apply
```

This means the run exists and is correct, but its identity tag is outdated —
typically because a new hash-included parameter was added without running the
migration scripts. The fix is a rehash, **not** retraining.

---

## Running the script

### Basic — check a specific config

```bash
uv run python scripts/01_dry_run_models.py \
    loss.rho=0.008 loss.topology=torus trial=0 \
    ${MLFLOW_TRACKING_URI:+mlflow.tracking_uri="$MLFLOW_TRACKING_URI"} \
    ${MLFLOW_ARTIFACT_LOCATION:+mlflow.artifact_location="$MLFLOW_ARTIFACT_LOCATION"} \
    ${MLFLOW_EXPERIMENT_NAME:+mlflow.experiment_name="$MLFLOW_EXPERIMENT_NAME"}
```

### With candidate filter — narrow the search

Use `+dry_filter.<key>=<value>` to add `params.<key> = '<value>'` clauses to
the MLflow candidate query. This is useful when you have many runs and want to
focus the diff on a relevant subset.

```bash
uv run python scripts/01_dry_run_models.py \
    loss.rho=0.008 loss.topology=torus trial=0 \
    +dry_filter.topology=torus +dry_filter.trial=0 \
    ${MLFLOW_TRACKING_URI:+mlflow.tracking_uri="$MLFLOW_TRACKING_URI"} \
    ${MLFLOW_EXPERIMENT_NAME:+mlflow.experiment_name="$MLFLOW_EXPERIMENT_NAME"}
```

`dry_filter` keys use **MLflow param leaf names** (e.g. `topology`, `rho`, `trial`)
— the same names visible in the MLflow UI params panel. They are **not** Hydra
config paths.

> **Note**: `dry_filter.*` keys are not pre-declared in the YAML config. Prefix
> each one with `+` (append syntax) so Hydra accepts them as new keys.

### Smoke-test that an existing config is idempotent

```bash
uv run python scripts/01_dry_run_models.py \
    trial=0 loss.rho=1.0 loss.topology=grid training.epochs=1 \
    ${MLFLOW_TRACKING_URI:+mlflow.tracking_uri="$MLFLOW_TRACKING_URI"} \
    ${MLFLOW_EXPERIMENT_NAME:+mlflow.experiment_name="$MLFLOW_EXPERIMENT_NAME"}
```

Expected output when the run exists:

```
============================================================
Training Dry Run
============================================================
  identity_hash : b18b28caf2c590f3
  cfg_hash      : 21ff29e11dcd2ac7
  trial         : 0 / seed: 100

  WOULD SKIP — FINISHED run already exists:
    run_id    : 827d7b4d834afb12...
    started   : 2024-01-15 10:30:22
============================================================
```

---

## Reading the diff output

```
─── run a1b2c3d4e5f6 (28/33 params match) ───
  Param               Proposed       Existing
  ─────────────────   ────────────   ────────────
  epochs*             200            1
  rho*                0.008          1.0
  seed*               100            109
  topology*           torus          grid

  * Parameter is part of the identity hash.
```

| Column | Meaning |
|---|---|
| `Param` | MLflow param key (leaf name, as logged) |
| `*` suffix | This param is part of `model_identity_fields` — a difference here caused the hash miss |
| `Proposed` | Value derived from the current resolved config |
| `Existing` | Value stored in the MLflow run |
| `<missing>` | The param was not logged in that run (schema mismatch or optional field) |

**If no params differ but the run wasn't matched**, either:
- The run has a stale identity hash → see the `!!!` banner above
- A hashed field is not individually logged (unlikely but possible after schema changes)

---

## Params that are part of the identity hash

All params whose config path falls under `model.*`, `loss.*`, `dataset.*`, or
`training.*` are identity-included, as well as the top-level fields
`schema_version`, `trial`, and `seed`. The script marks these with `*`
automatically by calling `model_identity_fields(cfg, seed)` at runtime — no
hardcoded list.

See [`idempotency.md`](idempotency.md) §2 and [`config_system.md`](config_system.md)
§3 for the complete definition.

---

## What to do when you find the problem

| Diff shows | Root cause | Fix |
|---|---|---|
| `rho*` or `topology*` differ | Config override not applied | Double-check the CLI override spelling |
| `seed*` differs | Different `trial` value → different auto-seed | Pass the correct `trial=N` |
| `epochs*` differs | Default changed in YAML | Add explicit `training.epochs=N` or update the YAML |
| `!!!  STALE IDENTITY HASH !!!` | New hash-included param added without migration | Run `uv run scripts/migrations/rehash_identities.py --apply` |
| `<missing>` in Existing | Old run predates a param | Run the param backfill migration; see [`contributing.md`](contributing.md) §11 |
| No diff, no stale banner | Hash-included but unlogged field differs | Check `schema_version` tag; may need a full migration |

---

## Keeping the proposed param dict in sync

`_build_proposed_params()` in `01_dry_run_models.py` mirrors lines 289–336 of
`01_train_models.py`. If a new loggable parameter is added to the training script,
**both** must be updated together. The function contains a comment marking this
dependency.

A mismatch between the two param dicts would not break the dry run (scores would
simply be slightly off), but it would reduce the usefulness of the diff.

---

## Related docs

- [`idempotency.md`](idempotency.md) — identity hash system, registry, migration semantics
- [`config_system.md`](config_system.md) — Hydra config groups, hash inclusion rules
- [`contributing.md`](contributing.md) — safe change procedures and migration checklists
