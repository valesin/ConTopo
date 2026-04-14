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
