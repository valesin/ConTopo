# TopoLoss Parameter Migration Guide

## Why This Document Exists

Adding new fields under `loss.*` changes model identity because `loss.*` is included
in the model identity wildcard. Without migration, existing FINISHED model runs keep
stale identity hashes, and idempotency checks can re-train already-computed configs.

This document records:
- what changed,
- the migration defaults that preserve historical behaviour,
- and the exact migration + verification commands.

---

## New Conditional Params: TopoLoss Loss Type

Three fields were added to `LossConfig` for `loss.topography_type=topoloss`:

| Param | Config key | Migration default | Previous behaviour |
|---|---|---|---|
| TopoLoss factor H | `loss.topoloss_factor_h` | `"None"` | N/A — old runs used `ws` or `global`, never external topoloss |
| TopoLoss factor W | `loss.topoloss_factor_w` | `"None"` | N/A — old runs used `ws` or `global`, never external topoloss |
| TopoLoss scale | `loss.topoloss_scale` | `"None"` | N/A — old runs used `ws` or `global`, never external topoloss |

All three are **conditional fields**:
- if `loss.topography_type=topoloss`, all three must be explicitly set,
- otherwise all three must remain `None`.

This is enforced by `src/config/validation.py` at startup.

### Why migration defaults are `"None"`

Backfilling numeric values (for example `8.0, 8.0, 1.0`) would misrepresent what
old runs actually used. Existing runs did not execute with external topoloss, so
`"None"` is the only faithful backfill value.

---

## Migration Commands

Run in this order for every affected experiment.

### Step 1 — Backfill new loss params

```bash
# Dry-run first:
uv run scripts/migrations/backfill_params.py \
    --spec scripts/migrations/specs/topoloss_loss_params.yaml \
    --experiment "${MLFLOW_EXPERIMENT_NAME}" \
    ${MLFLOW_TRACKING_URI:+--tracking-uri "$MLFLOW_TRACKING_URI"}

# Apply:
uv run scripts/migrations/backfill_params.py \
    --spec scripts/migrations/specs/topoloss_loss_params.yaml \
    --experiment "${MLFLOW_EXPERIMENT_NAME}" --apply \
    ${MLFLOW_TRACKING_URI:+--tracking-uri "$MLFLOW_TRACKING_URI"}
```

> Remote tracking servers: pass `--tracking-uri` explicitly.
> `backfill_params.py` defaults to `sqlite:///outputs/mlflow.db`.

### Step 2 — Recompute identity hashes

```bash
# Dry-run first:
uv run scripts/migrations/rehash_identities.py \
    --experiment "${MLFLOW_EXPERIMENT_NAME}" \
    ${MLFLOW_TRACKING_URI:+--tracking-uri "$MLFLOW_TRACKING_URI"}

# Apply:
uv run scripts/migrations/rehash_identities.py \
    --experiment "${MLFLOW_EXPERIMENT_NAME}" --apply \
    ${MLFLOW_TRACKING_URI:+--tracking-uri "$MLFLOW_TRACKING_URI"}
```

Backfill first, then rehash.

---

## Verification

### 1) Idempotency check for an old config

```bash
uv run scripts/01_train_models.py training.epochs=1 trial=0 loss.rho=0 loss.topology=grid
# Expected: existing run is detected and skipped.
```

### 2) Smoke-test the new topoloss mode

```bash
uv run scripts/01_train_models.py training.epochs=1 \
    loss=topoloss \
    loss.topoloss_factor_h=8.0 \
    loss.topoloss_factor_w=8.0 \
    loss.topoloss_scale=1.0
```

---

## References

- `scripts/migrations/specs/topoloss_loss_params.yaml`
- `src/config/structured.py` (`LossConfig`)
- `src/config/validation.py` (`validate_training_config`)
- `src/config/hash.py` (`model_identity_fields`, `identity_hash`)
- `docs/contributing.md` §10.2 / §10.4
