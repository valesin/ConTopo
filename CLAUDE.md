# ConTopo — Orientation for Claude

ConTopo is a Hydra + PyTorch + MLflow research pipeline for topographic
regularization experiments on CIFAR-10 / ImageNet100.

## Start here

- **[`README.md`](README.md)** — user manual: quick start, dataset switching,
  training backends (torch/FFCV), running experiments under different
  conditions, Docker/SkyPilot recipes.
- **[`docs/README.md`](docs/README.md)** — developer documentation index. Each
  doc has a distinct role; jump from the index.

## Key developer references (in `docs/`)

| Doc | When to read |
|---|---|
| [`architecture.md`](docs/architecture.md) | Runtime model, stage I/O, MLflow boundaries, invariants |
| [`config_system.md`](docs/config_system.md) | Hydra groups, hash inclusion, validation rules, adding a new parameter (three cases) |
| [`idempotency.md`](docs/idempotency.md) | Identity hashes, `IDEMPOTENCY_REGISTRY`, migration semantics, broken-hash recovery |
| [`telemetry_schema.md`](docs/telemetry_schema.md) | MLflow logging contract per run kind |
| [`contributing.md`](docs/contributing.md) | Safe change procedures, migration checklists, verification |
| [`ffcv_param_assumptions.md`](docs/ffcv_param_assumptions.md) | Worked example: a hash-included + conditional migration |
| [`analysis_guide.md`](docs/analysis_guide.md) | Notebook + MLflow analysis reference |

## Conventions to respect

- **Repository-first MLflow retrieval** — use
  `src/repositories/functional_run_repository.py`; never reintroduce ad-hoc
  finders in `src/mlflow_utils.py`.
- **Telemetry schema authority** — any logged param/tag/metric/artifact must
  appear in `TELEMETRY_SCHEMA` (`src/mlflow_schema_logger.py`). Add new
  entries to `"optional"` first.
- **Hash coherence** — fields in `training.*`, `model.*`, `loss.*`, `dataset.*`
  affect identity. Adding one requires the migration protocol in
  [`contributing.md`](docs/contributing.md) §10.2.
- **Conditional fields** — optional features default to `None`, never a
  fictitious numeric value; enforced at startup by `src/config/validation.py`.
- **Smoke-test only** — verification runs use `training.epochs=1`; never
  trigger full training without an explicit request.
