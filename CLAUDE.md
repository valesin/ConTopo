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
| [`training.md`](docs/training.md) | Training loop design: GradNormBalancer, MULTIRUN OOM, AMP, early stopping |
| [`ffcv_param_assumptions.md`](docs/ffcv_param_assumptions.md) | Worked example: a hash-included + conditional migration |
| [`topoloss_param_assumptions.md`](docs/topoloss_param_assumptions.md) | Worked example: TopoLoss loss-parameter migration |
| [`analysis_guide.md`](docs/analysis_guide.md) | Notebook + MLflow analysis reference |

## Notebook conventions (enforced style)

- **Two patterns** — Pattern A (raw `get_runs` → `varying_fields` → `mo.sql` filter) and
  Pattern B (`_for_groups` helpers → controls before load → Python filter). See
  [`analysis_guide.md`](docs/analysis_guide.md) §6 for the full spec.
- **matplotlib only** — no Altair or Plotly in new or updated notebooks.
  Use `plt.subplots(..., constrained_layout=True)`; never `fig.tight_layout()`.
- **UI controls after inspect** (Pattern A) — dropdowns/multiselects go in a cell *after*
  `varying_fields` so you know what's available before declaring what to vary.
- **UI controls before load** (Pattern B) — groups config and split dropdowns must
  precede the `get_X_results_for_groups(...)` call because they drive the query.
- **`mo.stop` guards** — every cell that can produce an empty result should call
  `mo.stop(condition, mo.callout(...))` before doing any further computation.

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
