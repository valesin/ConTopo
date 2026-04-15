# FFCV Param Assumptions & Migration Guide

## Why This Document Exists

Every new field added to `TrainingConfig` (which is hash-included) changes the
`identity_hash` for all existing model runs. Without migration, the training
script will not recognise existing runs as already-computed and will attempt to
re-run them.

This document records (a) what was hardcoded before each param existed, (b) the
migration default that preserves the old behaviour, and (c) the commands to run.

---

## Pre-existing Field — No Migration Needed

`training.scheduler` already existed in `TrainingConfig` and was already logged to
MLflow (`scheduler=none`) before this change. It was accepted by the config system
but never acted upon (no scheduler object was instantiated or stepped). This change
makes it functional.

---

## New Params: Training Recipe

These nine fields are new additions to `TrainingConfig`. All existing model runs
were trained with the behaviours listed in the "Previous behaviour" column.

| Param | Config key | Migration default | Previous behaviour |
|---|---|---|---|
| LR peak epoch | `training.lr_peak_epoch` | `None` | N/A — scheduler was always `none`; field was never used |
| Label smoothing | `training.label_smoothing` | `0.0` | `nn.CrossEntropyLoss()` — no smoothing |
| Blurpool | `training.use_blurpool` | `False` | Standard `MaxPool2d` + strided `Conv2d` |
| Selective weight decay | `training.optimizer_selective_wd` | `False` | WD applied uniformly to all parameters |
| Test-time augmentation | `training.lr_tta` | `False` | Single forward pass at validation |
| Progressive resolution (min) | `training.progressive_res_min` | `None` | Static image size from `dataset.image_size` |
| Progressive resolution (max) | `training.progressive_res_max` | `None` | Static image size from `dataset.image_size` |
| Progressive resolution ramp start | `training.progressive_res_start_ramp` | `None` | N/A — progressive resolution was never active; field was never used |
| Progressive resolution ramp end | `training.progressive_res_end_ramp` | `None` | N/A — progressive resolution was never active; field was never used |

**Design note:** `lr_peak_epoch`, `progressive_res_start_ramp`, and `progressive_res_end_ramp`
are *conditional fields* — they are only meaningful when their parent feature is active
(`scheduler=cyclic` or `progressive_res_min != null`, respectively). Their migration
default is `None` (not a numeric value) because old runs never used these features.
Backfilling a fictitious numeric value would misrepresent what actually ran. The config
validator (`src/config/validation.py`) enforces that these fields are set if and only if
their parent feature is active, catching misconfigurations at startup.

## New Params: Training Recipe (continued — hash-included, require migration)

These four fields were originally placed in `RuntimeConfig` during the FFCV integration
but were later moved to `TrainingConfig` because they directly affect the training data
pipeline and therefore affect the trained model.  All existing runs used the torch
backend and never wrote beton files, so their migration defaults are `"torch"` / `"None"`.

| Param | Config key | Migration default | Previous behaviour |
|---|---|---|---|
| Loading backend | `training.loading_backend` | `"torch"` | All existing runs used torch; hash was never sensitive to this |
| Beton max resolution | `training.beton.max_resolution` | `"None"` | N/A — torch runs never use beton format |
| Beton JPEG quality | `training.beton.jpeg_quality` | `"None"` | N/A — torch runs never use beton format |
| Beton compress probability | `training.beton.compress_probability` | `"None"` | N/A — torch runs never use beton format |

All four beton format fields are conditional: they must be `None` when
`loading_backend=torch` and must be explicitly set when `loading_backend=ffcv`.
The config validator (`src/config/validation.py`) enforces this at startup.

## New Params: Runtime (hash-excluded — no identity hash migration needed)

Only the beton storage directory remains in `RuntimeConfig`. It is purely operational
and does not affect what is stored in the beton file — only where it is stored on disk.

| Param | Config key | Default | Notes |
|---|---|---|---|
| Beton dir | `runtime.beton.dir` | `"outputs/betons"` | Hash-excluded; not logged |

## Installing the FFCV dependencies

The `ffcv` optional group installs three packages:

```bash
uv sync --group ffcv
```

| Package | Purpose |
|---|---|
| `ffcv` | Binary data loading and pipeline |
| `antialiased-cnns` | Blurpool (antialiased pooling) for `use_blurpool=true` |
| `cupy-cuda12x` | GPU-side normalisation kernel (`NormalizeImage` requires cupy) |

The cupy version must match the installed CUDA toolkit. `cupy-cuda12x` covers CUDA 12.x.
If you are on a different CUDA version, change the dependency accordingly.

---

## Migration Commands

Run in this order for **every affected MLflow experiment** before deploying this
change to production. The scripts are idempotent — safe to re-run.

### Step 1 — Backfill new training params

```bash
# Dry-run first (no writes — review PATCH/SKIP output):
uv run scripts/migrations/backfill_params.py \
    --spec scripts/migrations/specs/ffcv_training_params.yaml \
    --experiment <experiment_name>

# Apply (writes missing params to all FINISHED model runs):
uv run scripts/migrations/backfill_params.py \
    --spec scripts/migrations/specs/ffcv_training_params.yaml \
    --experiment <experiment_name> --apply
```

Adds all thirteen `training.*` params listed above to runs that do not already
have them. Runs that already have a param set are skipped unchanged. This includes
`loading_backend` and the three `beton_*` params added when those fields moved from
`RuntimeConfig` to `TrainingConfig`.

### Step 2 — Recompute identity hashes

```bash
# Dry-run first (prints old vs new hash for each affected run):
uv run scripts/migrations/rehash_identities.py --experiment <experiment_name>

# Apply (updates identity_hash tag on affected runs):
uv run scripts/migrations/rehash_identities.py --experiment <experiment_name> --apply
```

Recomputes `tags.identity_hash` for every FINISHED model run by downloading its
stored `config.yaml` artifact and merging it with the current `TrainingConfig`
defaults (via `_canonical_section()`). New fields automatically receive their
migration default values. Old field values are preserved.

### Why this order?

Param backfill (Step 1) ensures every run has visible params in the MLflow UI
before the hash tag is updated. The identity hash migration (Step 2) reads from
the stored config artifact — not from MLflow params — so it is technically
independent. Maintaining this order leaves the experiment in a consistent,
queryable state throughout.

---

## Verifying Success

After both migration steps, verify that idempotency is restored:

```bash
# Re-running with an existing config should hit the idempotency check and skip:
uv run scripts/01_train_models.py trial=0 loss.rho=0.0 loss.topology=torus
# Expected log line: "Run already exists, skipping." — training does NOT restart.
```

---

## FFCV pipeline implementation notes

These notes document behaviour that affects debugging or future maintenance:

**float16 pipeline:** FFCV image pipelines decode images onto GPU and normalise using
a cupy kernel, producing `float16` tensors. The training and validation loops cast
to `float32` (`.float()`) before the model forward pass to match model weight dtype.

**Label shape:** FFCV's `IntDecoder` + `ToTensor` produces labels of shape `[B, 1]`.
The training loop calls `.squeeze()` to reduce to `[B]` for `CrossEntropyLoss`.

**Eval crop ratio:** `CenterCropRGBImageDecoder` is created with a ratio capped at
`min(256/224, stored_size/image_size)` where `stored_size = min(max_resolution, image_size)`.
For small datasets like CIFAR-10 (stored at 32×32), this clamps to 1.0, preventing
an OpenCV assertion failure from requesting a crop region larger than the stored image.

**OneCycleLR `pct_start`:** When `epochs < lr_peak_epoch` (e.g. smoke-test runs with
`training.epochs=1`), `lr_peak_epoch` is clamped to `epochs - 1` before computing
`pct_start = lr_peak_epoch / epochs` to keep the value in `(0, 1)` as required.

---

## Reference

- `src/config/hash.py` — `identity_hash`, `model_identity_fields`, `EXCLUDED_KEYS`
- `src/config/structured.py` — `TrainingConfig`, `RuntimeConfig`, `BetonConfig`
- `src/config/validation.py` — `validate_training_config` (conditional field enforcement)
- `src/data/ffcv_pipelines.py` — FFCV augmentation pipeline builders
- `src/data/beton_writer.py` — on-demand beton generation
- `docs/idempotency.md` — full idempotency contract and migration protocol
