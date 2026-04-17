# Plan: `scripts/01_dry_run_models.py` — Training Idempotency Dry Run

## Context

Running `01_train_models.py` with a new config silently starts full training if no idempotency match is found. When a run is unexpectedly not skipped (hash miss), diagnosing the cause requires manually comparing params across MLflow runs.

This script provides a zero-side-effect inspection pass. It mirrors `01_train_models.py`'s config resolution **exactly** — same Hydra decorator, same `validate_training_config`, same `resolve_seed`, same `cfg_hash`, same `find_finished_model_run` — so the hash it computes is guaranteed to be the same hash the real script would compute.

The user also provides a filter expression (via `dry_filter.*` overrides) to select candidate runs from MLflow. The script then diffs the fully-resolved proposed params against each matched run's logged params.

---

## Design

### Key constraint: exact config resolution parity

The script must use the **same Hydra `@hydra.main` decorator, config path, and config name** as `01_train_models.py`. This ensures that all Hydra composition, interpolations, and defaults resolve identically. No hardcoded param extraction.

The proposed param dict is derived by taking `cfg` after full Hydra resolution and calling the same `schema_log_params` keys as the training script — but **not actually calling `schema_log_params`** (no MLflow write). Instead we build it from `cfg` directly using a helper that mirrors lines 289–336 of `01_train_models.py`.

### Filter params (new CLI surface)

A new Hydra config group `dry_filter` is introduced. Its keys map 1:1 to MLflow filter predicates. Example:

```bash
uv run python scripts/01_dry_run_models.py \
    loss.rho=0.008 loss.topology=torus trial=0 \
    dry_filter.topology=torus dry_filter.rho=0.008
```

`dry_filter.*` keys are translated to a single MLflow filter string:

```python
"tags.kind = 'model' and attributes.status = 'FINISHED' and params.topology = 'torus' and params.rho = '0.008'"
```

This is additive — if no `dry_filter` keys are given, all FINISHED model runs are fetched and ranked by similarity.

`dry_filter` is added to `EXCLUDED_KEYS` in `cfg_hash` (or simply never included in the hash) — it is a purely operational key.

---

## What the script does

1. **Setup** — same Hydra decorator/config path as `01_train_models.py`, same `validate_training_config`, `resolve_seed`, `cfg_hash`, `setup_mlflow`, `configure_run_repository`.
2. **Idempotency check** — `find_finished_model_run(cfg, seed)`.
   - **Match found** → print "WOULD SKIP" with `run_id`, `identity_hash`, `cfg_hash`, run start time. Done.
   - **No match** → print "WOULD TRAIN" and proceed to candidate search.
3. **Candidate search** — build MLflow filter from `dry_filter.*` overrides + `tags.kind = 'model' AND status = 'FINISHED'`. Fetch all matching runs via `search_runs(..., output_format="pandas")`. Rank by number of matching params.
4. **Diff output** — for the top-3 closest runs, print a two-column table of differing params only. Params that are part of `model_identity_fields` (i.e., everything in `training.*`, `model.*`, `loss.*`, `dataset.*` plus `schema_version`, `trial`, `seed`) are marked with `*`.

---

## Files

| File | Role |
|---|---|
| `scripts/01_dry_run_models.py` | Replace existing script |
| `conf/dry_filter/default.yaml` | New Hydra config group, all keys default to `null` |
| `conf/config.yaml` | Add `dry_filter: default` to defaults list |
| `src/config/hash.py` | Add `"dry_filter"` to `EXCLUDED_KEYS` |
| `scripts/01_train_models.py` | Reference only — no changes |
| `src/config/hash.py` | Reuse `model_identity_fields` to mark `*` params |
| `src/repositories/functional_run_repository.py` | Reuse `find_finished_model_run`, `search_runs`, `configure_run_repository` |
| `src/mlflow_utils.py` | Reuse `setup_mlflow`, `resolve_seed`, `validate_training_config` |

---

## Implementation detail

### `conf/dry_filter/default.yaml`

```yaml
# dry_filter — MLflow query params for 01_dry_run_models.py
# Set any key to filter the candidate run search.
# These keys do NOT affect the identity hash.
_target_: null  # prevents Hydra from treating this as a structured config
```

Each key the user sets here (e.g. `dry_filter.topology=torus`) becomes:
`params.topology = 'torus'` in the MLflow filter string.

### `conf/config.yaml` addition

```yaml
defaults:
  - dry_filter: default
  ...
```

### `src/config/hash.py` — `EXCLUDED_KEYS` addition

```python
EXCLUDED_KEYS = frozenset({
    ...,
    "dry_filter",   # <-- new
})
```

### Setup sequence (mirrors `01_train_models.py` lines 195–217)

```python
validate_training_config(cfg)
seed = resolve_seed(cfg); cfg.seed = seed
hash_val = cfg_hash(cfg)
setup_mlflow(cfg)
configure_run_repository(cfg.mlflow.tracking_uri, cfg.mlflow.experiment_name)
existing_run, model_identity_hash = find_finished_model_run(cfg, seed)
```

### Proposed param dict

Built inline from `cfg`, exactly mirroring lines 289–336 of `01_train_models.py`. `None` values are filtered out (matching `_clean_params` in the schema logger). This is the only place that mirrors the real script's param dict — if that dict changes, this script must be updated in sync.

### Identity param detection

```python
from src.config.hash import model_identity_fields, flatten_identity_section

identity_fields = set(model_identity_fields(cfg, seed).keys())
# e.g. {"schema_version", "trial", "seed", "model.arch", "loss.rho", ...}
```

The MLflow logged param name is the **leaf** (e.g. `rho`, not `loss.rho`). The mapping from MLflow key → identity is determined by checking if any `model_identity_fields` key ends with `.{mlflow_key}` or equals `mlflow_key`:

```python
def _is_identity_param(mlflow_key: str, identity_fields: set[str]) -> bool:
    return any(
        f == mlflow_key or f.endswith(f".{mlflow_key}")
        for f in identity_fields
    )
```

### MLflow filter from `dry_filter`

```python
from omegaconf import OmegaConf

dry = OmegaConf.to_container(cfg.dry_filter, resolve=True) or {}
clauses = ["tags.kind = 'model'", "attributes.status = 'FINISHED'"]
for k, v in dry.items():
    if v is not None:
        clauses.append(f"params.{k} = '{v}'")
filter_str = " and ".join(clauses)
```

### Diff display (top-3 by score)

```
─── run abc123... (28/30 params match) ───
Param                    Proposed       Existing
───────────────────────  ─────────────  ──────────────
rho*                     0.04           0.008
epochs*                  200            100

  * Parameter is part of the identity hash.
```

---

## Sample output

**Idempotent case:**
```
============================================================
Training Dry Run
============================================================
  identity_hash : abc123def4567890
  cfg_hash      : xyz987abc1234567
  trial         : 2 / seed: 102

  WOULD SKIP — FINISHED run already exists:
    run_id    : a1b2c3d4e5f6...
    started   : 2024-01-15 10:30:22
============================================================
```

**Non-idempotent case:**
```
============================================================
Training Dry Run
============================================================
  identity_hash : abc123def4567890
  cfg_hash      : xyz987abc1234567
  trial         : 2 / seed: 102

  WOULD TRAIN — no existing run matches this identity hash.

Searching for similar FINISHED model runs...  (12 found)

─── run a1b2c3d4e5f6 (28/30 params match) ───
Param               Proposed       Existing
──────────────────  ─────────────  ─────────────
rho*                0.04           0.008
epochs*             100            50

  * Parameter is part of the identity hash.
============================================================
```

---

## Constraints

- **No training, no MLflow writes** — read-only after `configure_run_repository`
- **No new abstractions** — all helpers from existing modules
- **Exact config resolution parity** — same Hydra decorator, same setup sequence as `01_train_models.py`

---

## Verification

```bash
# Should print WOULD SKIP for a run that already exists:
uv run python scripts/01_dry_run_models.py trial=0 loss.rho=1.0 loss.topology=grid training.epochs=1 model.arch=LinearSimpleCNN

# Should print WOULD TRAIN + diff for a novel config:
uv run python scripts/01_dry_run_models.py trial=99 loss.rho=999

# With explicit filter to narrow candidate search:
uv run python scripts/01_dry_run_models.py trial=0 loss.rho=0.008 \
    dry_filter.topology=torus dry_filter.trial=0
```
