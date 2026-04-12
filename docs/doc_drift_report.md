# Documentation Drift Report

## Scope

This report compares the current executable codebase against existing top-level documentation and state reports.

Primary truth sources used:

- `main.py`
- `conf/config.yaml`, `conf/pipeline/*.yaml`, `conf/*/default.yaml`
- `scripts/01_train_models.py` … `scripts/05_train_adapters.py`
- `src/repositories/functional_run_repository.py`
- `src/config/hash.py`
- `src/mlflow_schema_logger.py`
- `src/mlflow_utils.py`
- `notebooks/mlflow/mlflow_helpers.py`

## Confirmed drifts

### 1) Orchestrator CLI examples are stale in old README

Observed drift:
- Old examples mention `--skip-training` and `--from-step 3` style flags.

Current code reality:
- `main.py` accepts Hydra overrides, e.g. `pipeline.from_step=ensemble` and `+pipeline=small`.
- There is no `--skip-training` parser flag.

Impact:
- Copy/paste command failures for users.

Resolution:
- Rewrote README command section to match Hydra-based usage.

---

### 2) Legacy retrieval APIs are referenced in documentation

Observed drift:
- Old docs reference retrieval helpers in `src/mlflow_utils.py` (e.g., helper-style finder/query functions).

Current code reality:
- Retrieval is centralized in `src/repositories/functional_run_repository.py`.
- `src/mlflow_utils.py` now focuses on setup/logging/artifact utilities.

Impact:
- Developers add new retrieval logic in the wrong layer.

Resolution:
- New architecture docs explicitly enforce repository as retrieval SSOT.

---

### 3) Notebook/query examples call non-existent paths/functions

Observed drift:
- Old README examples use `src.config.notebook.setup_environment` and old artifact paths such as `inference_data/...`.

Current code reality:
- Active notebook helper is `notebooks/mlflow/mlflow_helpers.py`.
- Inference artifacts are logged under `inference/{split}_*.parquet|npz`.

Impact:
- Analysis onboarding breaks quickly.

Resolution:
- Added `ANALYSIS_GUIDE.md` with working retrieval patterns and current artifact locations.

---

### 4) Historical architectural state report contains invalid cleanup claims

Observed drift:
- `reports/system_state_20260412/current_architectural_state.md` claims migration/inspection scripts were deleted.

Current code reality:
- Some maintenance scripts still exist (for example `scripts/inspect_and_rehash_model_identity.py`, `scripts/mlflow_gc.py`).

Impact:
- Incorrect assumptions during maintenance and audit work.

Resolution:
- New docs avoid "deleted" claims and document present-state behavior only.

---

### 5) `docs/idempotency.md` describes pre-registry identity model

Observed drift:
- Document references legacy functions and identity semantics no longer aligned to current implementation.

Current code reality:
- `IDEMPOTENCY_REGISTRY` in `src/config/hash.py` is the source for required identity fields.
- `identity_hash` validates both unknown and missing fields.

Impact:
- High risk of introducing collisions or invalid assumptions.

Resolution:
- New `ARCHITECTURE.md` and `CONTRIBUTING_AND_UPDATING.md` anchor idempotency updates to the registry + tests.

---

### 6) `docs/config_system.md` overstates active structured-config runtime role

Observed drift:
- Document describes strict dataclass registration workflow as central runtime behavior.

Current code reality:
- Pipeline scripts use Hydra composition directly via YAML groups; runtime behavior depends on YAML + code paths in scripts.
- Structured config tests exist, but runtime scripts do not centrally depend on an explicit `register_configs()` bootstrap.

Impact:
- Developers may assume safeguards that are not in execution path.

Resolution:
- Updated docs focus on executable YAML/group contracts and step-level behavior.

## Additional notes

- `docs/telemetry_schema.md` is broadly aligned with runtime enforcement and remains useful.
- `docs/ADD_MODEL.md` and `docs/add_adapter.md` are lightweight and partially outdated in architecture details (notably adapter extension path); superseded by the new contribution guide.

## Deliverables produced from this report

- Rewritten `README.md`
- New `ARCHITECTURE.md`
- New `CONTRIBUTING_AND_UPDATING.md`
- New `ANALYSIS_GUIDE.md`
