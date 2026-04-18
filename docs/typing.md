# Typing status and roadmap

This document tracks the current typing strategy for ConTopo and where strict
coverage is intentionally deferred.

## Current coverage

The highest-impact runtime paths now have explicit return/parameter typing:

- Central TypedDicts in `src/types.py`:
  - `InferenceOutput`
  - `AnchorSpec`
  - `AnchorOutput`
- Inference and anchor APIs:
  - `src/inference.py`
  - `src/data/anchors.py`
- Core utility modules with concrete signatures:
  - `src/ensemble/accuracy.py`
  - `src/ensemble/selector.py`
  - `src/profiling/category_similarity.py`
  - `src/profiling/masking.py`
  - `src/profiling/diversity.py`
  - `src/config/validation.py`
  - `src/config/hash.py`
  - `src/config/notebook.py`
  - `src/losses/topographic.py`
  - `src/losses/balancer.py`
  - `src/networks/resnet18.py`
  - `src/networks/simple_cnn.py`
  - `src/networks/resnet34_imagenet.py`
  - `src/networks/heads.py`
  - `src/networks/adapter_registry.py`
  - `src/training/train_ce.py`
  - `src/repositories/functional_run_repository.py`
  - `src/repositories/functional_service_example.py`
  - `src/data/loaders.py`
  - `src/mlflow_schema_logger.py`
  - `src/mlflow_utils.py`

## Intentional `Any` boundaries

Some integrations remain permissive because upstream libraries are weakly typed
or have unstable stub quality:

- `mlflow` entity/query boundaries
- `ffcv` data pipeline boundaries
- dynamic factory dispatch patterns where return type depends on runtime config

This keeps strictness focused on internal logic while avoiding high-churn type
noise at third-party boundaries.

## basedpyright policy

`basedpyright` is enabled in basic mode for `src/` with targeted enforcement:

- Enabled as errors:
  - `reportMissingParameterType = "error"`
  - `reportReturnType = "error"`
  - `reportArgumentType = "error"`
- Enabled as warnings (staged hardening):
  - `reportAttributeAccessIssue = "warning"`
- Intentionally suppressed for now:
  - `reportPrivateImportUsage = "none"`
  - missing imports/type stubs for optional or weakly typed boundaries
  - unknown-* and generic `Any` diagnostics

Rationale: catch call/return/signature regressions in first-party code while
holding a documented exception for third-party stub-export behavior.

### Exception note: `reportPrivateImportUsage`

`reportPrivateImportUsage` was evaluated during hardening and produced a large
volume of diagnostics against standard PyTorch/MLflow usage patterns (for
example `torch.tensor`, `torch.softmax`, `mlflow.tracking.*`) due to upstream
stub export surfaces rather than first-party design errors.

Treating those as actionable would require broad, invasive rewrites or local
shadow stubs for core libraries, which is disproportionate and brittle. For
that reason, this check is intentionally disabled as an explicit exception.

## Pre-commit enforcement

Typing checks are enforced via local pre-commit hook in `.pre-commit-config.yaml`:

- hook id: `basedpyright`
- command: `uv run basedpyright`
- scope: repository-wide (`pass_filenames: false`)

This ensures typed-check parity between local workflows and CI-style checks.

## Current verification state

Latest validation results:

1. `uv run basedpyright src/` → **0 errors** (warnings remain by staged policy)
2. `uv run pytest` → **all tests passing**
3. `pre-commit run basedpyright --all-files` → **passing**

## Path to ~95% typed coverage

Reaching near-complete strict coverage would require a dedicated hardening pass:

1. Add stable typing wrappers/stubs for MLflow query and run objects.
2. Introduce explicit protocol-based interfaces for FFCV loader artifacts.
3. Add overloads for config-driven factory functions in model/loss/metric registries.
4. Replace remaining broad dict payloads with TypedDict/dataclass contracts.
5. Raise basedpyright strictness incrementally (promote staged warnings to errors).

Estimated effort: medium-to-high (roughly 2-4 focused engineering days), mostly
in external-boundary wrappers and factory dispatch typing.
