# Contributing and Updating

This guide describes safe update procedures for the current ConTopo architecture.

## 1. Non-negotiable rules

1. Use `src/repositories/functional_run_repository.py` for MLflow retrieval.
2. Keep telemetry logging aligned with `src/mlflow_schema_logger.py`.
3. Keep identity semantics aligned with `src/config/hash.py` (`IDEMPOTENCY_REGISTRY`).
4. Prefer config-driven behavior through `conf/*` groups and `conf/pipeline/*` steps.

## 2. Standard developer loop

1. Implement the minimal code change.
2. Update config/docs as needed.
3. Run focused tests first, then broader tests as needed.
4. Validate one representative stage command.

## 3. Adding/changing MLflow retrieval logic

Do:

- add or update repository functions in `src/repositories/functional_run_repository.py`.
- call `configure_run_repository(...)` in scripts before repository calls.

Do not:

- reintroduce ad-hoc run finder wrappers in `src/mlflow_utils.py`.

## 4. Adding telemetry fields

If you add a new logged value for a run kind:

1. Update `TELEMETRY_SCHEMA` in `src/mlflow_schema_logger.py`.
2. Log the field in the stage script using schema wrappers.
3. Run a focused execution to ensure validation passes.

If a value is required by the run contract, place it in the `required` list; otherwise use `optional`.

## 5. Changing idempotency semantics

If semantic outputs change, update identity definition:

1. edit `IDEMPOTENCY_REGISTRY` in `src/config/hash.py`.
2. update call sites computing `identity_hash(...)` for that kind.
3. add/update tests (especially `tests/test_idempotency_registry.py`).

For model-level identity behavior, also validate `cfg_hash` expectations (`tests/test_cfg_hash.py`).

## 6. Adding a new pipeline stage

1. create `scripts/<NN>_*.py` stage script.
2. choose run `kind` and add schema entry in `TELEMETRY_SCHEMA`.
3. add identity definition in `IDEMPOTENCY_REGISTRY`.
4. register the stage in `conf/pipeline/default.yaml` (and optionally `small.yaml`).
5. add targeted tests.

## 7. Updating Hydra config safely

When adding a new config parameter:

1. add field in the appropriate `conf/<group>/*.yaml`.
2. ensure script reads/uses it.
3. decide whether it should affect identity:
   - model semantic change: include via existing included groups for `cfg_hash`.
   - stage semantic change: include in that stage's `identity_hash` fields.
4. add tests for hash sensitivity/insensitivity as appropriate.

## 8. Analysis helper updates

For notebook-facing retrieval/format conveniences, update:

- `notebooks/mlflow/mlflow_helpers.py`

Keep this layer focused on analysis ergonomics, not pipeline control flow.

## 10. Adding a new dataset

The pipeline is dataset-agnostic. Adding support for a new dataset requires touching
exactly two source files and creating a few config files — no pipeline logic changes.

### 10.1 Register the loader factory (`src/data/loaders.py`)

Add a factory function with this signature:

```python
def _<name>_factory(root: str, train: bool, transform, download: bool = False):
    """Return a torchvision-style Dataset with a .targets list of int labels."""
    ...
```

For datasets available in torchvision (e.g. CIFAR-100, STL-10):
```python
def _cifar100_factory(root, train, transform, download=True):
    with contextlib.redirect_stdout(io.StringIO()):
        return datasets.CIFAR100(root=root, train=train, transform=transform, download=download)
```

For custom datasets stored on disk as ImageFolder:
```python
def _mydataset_factory(root, train, transform, download=False):
    subset = "train" if train else "val"
    return datasets.ImageFolder(root=os.path.join(root, "mydataset", subset), transform=transform)
```

Then register it:
```python
_DATASET_FACTORIES["<name>"] = _<name>_factory
DATASET_NUM_CLASSES["<name>"] = <num_classes>
```

### 10.2 Create the dataset config (`conf/dataset/<name>.yaml`)

All required fields are already defined in the config schema — no `structured.py` changes needed:

```yaml
name: <name>
image_size: <H>           # e.g. 32 for CIFAR, 224 for ImageNet-scale
in_channels: 3
mean: [R, G, B]
std:  [R, G, B]

split:
  strategy: first_n_per_class
  val_per_class: <N>      # N × num_classes samples reserved for validation

transforms:
  preset: <preset_name>   # must exist in src/data/transforms.py
```

### 10.3 Add a transform preset if needed (`src/data/transforms.py`)

If the dataset needs a distinct augmentation strategy (e.g. different crop size or
a Resize+CenterCrop eval pipeline), add a named preset function and register it in
`_PRESETS`. Follow the existing versioning convention (e.g. `mydata_v1`).

Preset names are included in `cfg_hash`, so changing a preset's behaviour requires
creating a new version (`mydata_v2`) rather than modifying the existing one.

### 10.4 Create supporting configs

- **`conf/profiling/<name>.yaml`** — if the default anchor-per-class count (100) is
  inappropriate (e.g. 100 × 100 classes = 10 000 anchors for a 10 000-image test set).
  Scale down: `per_class: 10` gives 1 000 total for 100-class datasets.

- **`conf/model/<name>.yaml`** — if the dataset requires a different backbone
  (e.g. `LinearResNet34` with `arch: LinearResNet34` for large-image datasets).

- **`conf/sweeps/training_rho_<name>.yaml`** — training sweep with
  `override /dataset: <name>`, `override /model: <name>`, and dataset-specific
  training hyperparameters (batch_size, lr, epochs).

### 10.5 MLflow experiment isolation

Always use a distinct `mlflow.experiment_name` per dataset:

```bash
python main.py +sweeps=training_rho_<name>
# experiment_name is set inside the sweep YAML
```

This prevents ensemble discovery (`conf/groups/default.yaml` uses `filter: {}`)
from mixing models trained on different datasets within the same experiment.

### 10.6 Checklist

- [ ] Factory function added to `_DATASET_FACTORIES` in `src/data/loaders.py`
- [ ] Class count added to `DATASET_NUM_CLASSES` in `src/data/loaders.py`
- [ ] `conf/dataset/<name>.yaml` created with all 7 required fields
- [ ] Transform preset exists or new one added in `src/data/transforms.py`
- [ ] `conf/profiling/<name>.yaml` created (if anchor count needs scaling)
- [ ] `conf/model/<name>.yaml` created (if architecture differs)
- [ ] `conf/sweeps/training_rho_<name>.yaml` created with `mlflow.experiment_name`
- [ ] Smoke run completes: `python scripts/01_train_models.py dataset=<name> model=<name> training.epochs=1 trial=0`

## 11. Adding new configuration parameters

Every new parameter requires at least two updates regardless of type:
1. **YAML + structured config** — the value must be representable in the config system.
2. **Telemetry schema** — if the param is logged to MLflow, it must appear in
   `TELEMETRY_SCHEMA` in `src/mlflow_schema_logger.py`. Add it to `"optional"` so
   existing runs that pre-date the field still pass telemetry validation.

Beyond those two, the protocol depends on three properties of the new parameter:

---

### 11.1 Decision framework

Answer these questions in order:

**Q1: Does the parameter affect the trained model** (weights, training data, loss
computation, optimiser behaviour)?

- **Yes** → place in a hash-included config group (`training.*`, `model.*`, `loss.*`,
  `dataset.*`). Changing this param creates a semantically different model.
  → **Migration is mandatory** (§11.3).
- **No** → place in a hash-excluded group (`runtime.*`, `execution.*`, etc.). The
  param controls how a run executes, not what it produces.
  → **Migration is optional** but recommended for observability (§11.4).

**Q2: Is the parameter only meaningful when another setting is active** (e.g. a
scheduler-specific knob, or a format setting that only applies when a certain backend
is selected)?

- **Yes** → it is a **conditional field**. It must be `None` when its parent is
  inactive, and explicitly set when it is active. Requires validation rules (§11.5).
- **No** → it is an **unconditional field** with a concrete default.

These questions are independent. A conditional field can be hash-included (e.g.
`lr_peak_epoch`) or hash-excluded. An unconditional field can be either too.

---

### 11.2 Hash-included parameters (mandatory migration)

New fields in hash-included config groups (`training.*`, `model.*`, `loss.*`,
`dataset.*`) change the `identity_hash` for **all existing model runs**.  Without
migration, the training script will not recognise existing runs as already-computed
and will re-train them.

#### Step A — Write the param assumptions document first

Create `docs/<feature>_param_assumptions.md` **before touching any code**:
- What was previously hardcoded for this param (the behaviour before it existed)?
- What is the migration default that faithfully preserves the old behaviour?

See `docs/ffcv_param_assumptions.md` as the canonical example.

#### Step B — Write migration scripts

Two scripts are needed:

1. **Param backfill** — write a spec file `scripts/migrations/specs/<feature>.yaml`
   listing each new param and its migration default, then run the generic script:
   ```bash
   uv run scripts/migrations/backfill_params.py \
       --spec scripts/migrations/specs/<feature>.yaml \
       --experiment <experiment_name> [--apply]
   ```
   The script is idempotent: runs that already have a param set are skipped.
   See `scripts/migrations/specs/ffcv_training_params.yaml` as the canonical example.

2. **Identity hash rehash** (`scripts/migrations/rehash_identities.py`) — recomputes
   `tags.identity_hash` for every FINISHED model run by downloading its stored
   `config/resolved_config.yaml` artifact and merging it with the current struct
   defaults via `_canonical_section()`. New fields receive their defaults automatically.

Run param backfill first, then identity hash rehash. See `docs/ffcv_param_assumptions.md`
for the rationale.

#### Step C — Add the field

1. Add to the appropriate dataclass in `src/config/structured.py` with the correct
   default (concrete value for unconditional; `None` for conditional).
2. Add to the corresponding `conf/<group>/default.yaml` with a comment explaining
   the dependency condition or the migration rationale.
3. Wire up the logic in the relevant script(s).
4. Log the param in the `schema_log_params` call in the script.
5. Add to `"optional"` in `TELEMETRY_SCHEMA` in `src/mlflow_schema_logger.py`.

#### Checklist (hash-included)

- [ ] `docs/<feature>_param_assumptions.md` written
- [ ] `scripts/migrations/specs/<feature>.yaml` written; backfill dry-run reviewed
- [ ] `scripts/migrations/rehash_identities.py` dry-run reviewed
- [ ] Struct field added with correct default in `src/config/structured.py`
- [ ] `conf/<group>/default.yaml` updated with comment
- [ ] If conditional: validation rules added to `src/config/validation.py` (see §11.5)
- [ ] Script wires up the feature and logs the param
- [ ] `TELEMETRY_SCHEMA` updated (`"optional"` list)
- [ ] Migration scripts applied to all affected experiments
- [ ] Idempotency smoke-check: re-run existing config → "already FINISHED, skipping"

---

### 11.3 Hash-excluded parameters (observability migration, optional)

New fields in hash-excluded groups (`runtime.*`, `execution.*`, etc.) do **not** break
idempotency — existing runs still have correct identity hashes. Migration is not
required for correctness.

However, backfilling the migration default is still **recommended** for observability:
it allows you to query MLflow for "what value did this param have on older runs?" and
to see a consistent set of params across all runs in the UI.

The procedure is the same as §11.2 Steps A–C, except:
- No identity hash rehash is needed (skip the second migration script).
- Document the old hardcoded behaviour in `docs/<feature>_param_assumptions.md`
  under a "Runtime (hash-excluded)" section.

#### Checklist (hash-excluded)

- [ ] Struct field added with correct default in `src/config/structured.py`
- [ ] `conf/<group>/default.yaml` updated
- [ ] If conditional: validation rules added to `src/config/validation.py` (see §11.5)
- [ ] Script wires up the feature and logs the param
- [ ] `TELEMETRY_SCHEMA` updated (`"optional"` list)
- [ ] *(Optional)* Param backfill script written and applied for observability
- [ ] *(Skip)* No identity hash rehash needed

---

### 11.4 Conditional parameters

A parameter is conditional when it is only meaningful if a parent feature is active
(e.g. `lr_peak_epoch` is only used when `scheduler=cyclic`; beton format fields are
only used when `loading_backend=ffcv`).

**Rules for conditional fields — regardless of hash status:**

- Type as `Optional[T] = None` in the struct (`src/config/structured.py`).
- Use `null` as the default in the YAML (`conf/<group>/default.yaml`).
- Migration default must be `"None"` (not a fictitious numeric value). Backfilling
  a made-up value misrepresents what the run actually did and pollutes the hash.
- Add validation rules to `src/config/validation.py` that:
  - Error when the parent feature is active but the conditional field is `None`.
  - Error when the conditional field is set but the parent feature is inactive
    (orphaned field).

See §5 of `docs/config_system.md` for the full table of current conditional fields
and the validation rules they enforce.

## 9. Suggested verification commands

```bash
pytest tests/test_idempotency_registry.py
pytest tests/test_cfg_hash.py
pytest tests/test_hydra_config.py
```

Run a smoke pipeline after larger changes:

```bash
python main.py pipeline=small
```
