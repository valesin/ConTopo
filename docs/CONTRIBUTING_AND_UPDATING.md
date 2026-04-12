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
