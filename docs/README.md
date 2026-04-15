# ConTopo Documentation

**For users** (running experiments) → start at the top-level [`README.md`](../README.md).

**For developers** (extending the code) — the docs below each have a distinct,
non-overlapping role:

| Doc | Role |
|---|---|
| [`architecture.md`](architecture.md) | Runtime architecture, stage flow, MLflow boundaries |
| [`config_system.md`](config_system.md) | Hydra groups, hash inclusion, validation rules, adding parameters |
| [`idempotency.md`](idempotency.md) | Identity hashes, `IDEMPOTENCY_REGISTRY`, migration semantics |
| [`telemetry_schema.md`](telemetry_schema.md) | MLflow logging contract, run kinds, required vs optional |
| [`contributing.md`](contributing.md) | Safe change procedures + migration protocols |
| [`ffcv_param_assumptions.md`](ffcv_param_assumptions.md) | Worked example: FFCV training recipe migration |
| [`analysis_guide.md`](analysis_guide.md) | Notebook + MLflow analysis reference |

## Common tasks — where to start

**Adding a new config parameter?**
Read [`config_system.md`](config_system.md) §"Adding a new parameter — three
cases" to decide which case applies, then follow the relevant migration
checklist in [`contributing.md`](contributing.md).

**Changing an existing idempotency rule (what makes a run unique)?**
See [`idempotency.md`](idempotency.md) §5 for the migration protocol.

**Adding a new pipeline stage?**
See [`contributing.md`](contributing.md) §6 and
[`telemetry_schema.md`](telemetry_schema.md) for declaring the run kind.

**Writing a notebook that queries MLflow?**
Use [`analysis_guide.md`](analysis_guide.md) — `mlflow_helpers.py` inventory
and query patterns.

## Filename convention

All filenames under `docs/` are `snake_case.md` (all lowercase, underscores
between words). Do not introduce mixed-case or `PascalCase` filenames.
