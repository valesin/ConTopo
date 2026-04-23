# ConTopo Implementation Handoff Plan

This document is an implementation handoff for another agent.  
Repository root: `/home/runner/work/ConTopo/ConTopo`

---

## 1) Goal and Scope

This plan consolidates the deep analysis of:

- `src/` runtime logic
- `scripts/` runnable pipeline stages and migrations
- `docs/` architecture/config/idempotency/telemetry documentation
- `/home/runner/work/ConTopo/ConTopo/README.md`
- `/home/runner/work/ConTopo/ConTopo/notebooks/mlflow/mlflow_helpers.py`

It defines:

1. What to implement for **activation maps** as a new run kind.
2. How to redesign **`mlflow_helpers.py`** with a simpler standard interface.
3. What to formalize for a reusable pipeline abstraction.
4. Coherence issues that should be fixed or explicitly decided.

---

## 2) Current Architecture Snapshot (What the implementing agent must know)

### 2.1 Pipeline model

- Orchestrator: `/home/runner/work/ConTopo/ConTopo/main.py`
- Stage graph source of truth: `/home/runner/work/ConTopo/ConTopo/conf/pipeline/*.yaml`
- Main stages:
  - `01_train_models.py` (`kind=model`)
  - `02_cache_inference.py` (`kind=inference`)
  - `03_compute_profiles.py` (`kind=category_similarity_profile`)
  - `03b_compute_diagnostics.py` (`kind=diagnostics`)
  - `04_run_ensemble.py` (`kind=ensemble`)
  - `04b_compute_diversity.py` (`kind=diversity`)
  - `04c_compute_consistency.py` (`kind=consistency`)
  - `05_train_adapters.py` (`kind=metalearner`)

### 2.2 Required cross-cutting contracts

Any new run type/stage must be synchronized across:

1. **Idempotency registry**  
   `/home/runner/work/ConTopo/ConTopo/src/config/hash.py` (`IDEMPOTENCY_REGISTRY`)
2. **Telemetry schema**  
   `/home/runner/work/ConTopo/ConTopo/src/mlflow_schema_logger.py` (`TELEMETRY_SCHEMA`)
3. **Pipeline stage config**  
   `/home/runner/work/ConTopo/ConTopo/conf/pipeline/*.yaml`
4. **Script run semantics**  
   `execution.force`, `find_finished_identity_run`, and artifact logging pattern.
5. **Analysis helper layer**  
   `/home/runner/work/ConTopo/ConTopo/notebooks/mlflow/mlflow_helpers.py`

### 2.3 MLflow retrieval boundary

- Retrieval SSOT: `/home/runner/work/ConTopo/ConTopo/src/repositories/functional_run_repository.py`
- `src/mlflow_utils.py` should not become a second retrieval ownership layer.

### 2.4 Config and identity behavior

- Model identity includes: `schema_version`, `trial`, `seed`, plus `model.*`, `loss.*`, `dataset.*`, `training.*`.
- `runtime`, `mlflow`, `execution`, `groups`, etc. are hash-excluded for model identity.
- Conditional fields should default to `None` when inactive and be validated by `/home/runner/work/ConTopo/ConTopo/src/config/validation.py`.

---

## 3) Workstream A — Add `activation_maps` as a new run kind

## A1. Decisions to make before coding

1. **Stage placement decision**
   - Option 1: new script `03c_compute_activation_maps.py` (recommended).
   - Option 2: fold into `03b_compute_diagnostics.py` as another diagnostic metric.
   - Decision criteria:
     - lifecycle independence
     - artifact volume and compute cost
     - telemetry schema cleanliness

2. **Identity granularity decision**
   - Must define identity fields to avoid collisions/over-fragmentation.
   - Candidate minimum fields:
     - `parent_run_id`
     - `split`
     - `layer_spec` (or equivalent)
     - `activation_map_method`
   - Optional identity fields if output changes:
     - aggregation mode
     - class conditioning
     - normalization mode

3. **Source of activations decision**
   - Hook model forward pass from `e2e_best` (richer, costlier, backbone-coupled), or
   - derive from cached artifacts (cheaper, but potentially insufficient for true activation maps).

4. **Artifact contract decision**
   - Define exact artifact paths and formats (e.g., `activation_maps/*.pt`, summaries).
   - Decide whether to log tabular metadata + tensor artifacts together.

5. **Pipeline inclusion decision**
   - Add to production pipeline only, or also to `pipeline=small`.

## A2. Required implementation touchpoints

At minimum:

1. `/home/runner/work/ConTopo/ConTopo/src/config/hash.py`
   - Add `activation_maps` entry in `IDEMPOTENCY_REGISTRY`.

2. `/home/runner/work/ConTopo/ConTopo/src/mlflow_schema_logger.py`
   - Add `TELEMETRY_SCHEMA["activation_maps"]` with params/tags/metrics/artifacts.
   - Add new keys initially to `"optional"` where backward compatibility is needed.

3. New stage script (recommended)
   - `/home/runner/work/ConTopo/ConTopo/scripts/03c_compute_activation_maps.py`
   - Pattern must match existing stage behavior:
     - resolve parent model run
     - compute step `identity_hash`
     - skip when FINISHED and not `execution.force`
     - run + log schema-allowed params/tags/metrics/artifacts

4. Pipeline config updates
   - `/home/runner/work/ConTopo/ConTopo/conf/pipeline/default.yaml`
   - optional: `/home/runner/work/ConTopo/ConTopo/conf/pipeline/small.yaml`

5. Analysis helper support
   - Add retrieval and artifact loaders in `mlflow_helpers.py` redesign (Workstream B).

6. Documentation updates
   - `/home/runner/work/ConTopo/ConTopo/README.md` pipeline script list
   - `/home/runner/work/ConTopo/ConTopo/docs/architecture.md`
   - `/home/runner/work/ConTopo/ConTopo/docs/telemetry_schema.md`
   - `/home/runner/work/ConTopo/ConTopo/docs/idempotency.md`
   - `/home/runner/work/ConTopo/ConTopo/docs/analysis_guide.md`

## A3. Open questions to resolve with stakeholder

1. What exact semantic object is “activation map” in this project (layer activations, CAM/Grad-CAM, or another representation)?
2. Which layers are in-scope across backbones (`LinearResNet18`, `FinetuneResNet34`, `ScratchResNet34`, `LinearSimpleCNN`)?
3. Is split fixed (`test`) or configurable (`test|val|train`)?
4. Required output consumers: notebook only, or downstream pipeline stages?
5. Should activation maps be model-level only, or later extended to ensemble-level runs?

---

## 4) Workstream B — Redesign `notebooks/mlflow/mlflow_helpers.py`

Target file: `/home/runner/work/ConTopo/ConTopo/notebooks/mlflow/mlflow_helpers.py`

## B1. Current issues to fix

1. Many kind-specific list wrappers (`get_*_list`) conflict with desired generic API.
2. Mixed return ecosystems (Polars + pandas) increases friction.
3. Some function parameters are vestigial/misleading (`experiment_name` present but not used in some helpers).
4. Download/load helpers are not uniformly run_id-centric for simplest caller flow.

## B2. Required redesign outcomes

1. **Single list API by kind**
   - One function that takes run `kind` and returns full DataFrame.
   - No kind-specific list wrappers as primary interface.

2. **Simple artifact preload helpers by run_id**
   - Helpers should accept explicit `run_id`, load internally, and return ready data.
   - Caller should not manage low-level download logic.

3. **Maintain workflow documentation**
   - Keep high-clarity docstrings and module-level “how to use” sections.
   - Retain practical notebook usage guidance.

4. **Avoid unnecessary abstraction**
   - Keep file small and direct; avoid class hierarchy unless strictly needed.

## B3. Recommended API shape

1. Generic retrieval:
   - `get_runs(kind: str, status: str = "FINISHED", output="pandas") -> DataFrame`

2. Targeted run query:
   - `get_run_by_identity(kind: str, **identity_fields)` or narrow convenience helpers only where repeated.

3. Artifact loaders (run_id-first):
   - `load_inference_artifacts(run_id, split="test")`
   - `load_profile_artifacts(run_id, split="test", similarity_metric="cosine")`
   - `load_adapter_inputs(run_id, behaviour_input_hash=None)`
   - potential future: `load_activation_maps(run_id, ...)`

## B4. Migration strategy

1. Introduce new generic APIs first.
2. Keep old names as thin compatibility wrappers (temporary), with explicit deprecation docstrings.
3. Update notebooks gradually.
4. Remove wrappers after migration window.

---

## 5) Workstream C — Toward reusable ML/AI pipeline abstraction

## C1. Reusable core candidates

Good portability candidates:

- idempotency primitives (`identity_hash`, canonical field flattening)
- telemetry contract validation pattern (`start_run` + schema validation)
- MLflow repository access boundary (`search_runs`, `find_finished_identity_run`)
- generic stage skip/run pattern (`execution.force`, FINISHED hit)
- artifact cache-safe loading utility

## C2. Project-coupled areas

Currently tightly coupled to ConTopo:

- hardcoded run kinds and naming
- script numbering/stage naming conventions
- specific config key paths and assumptions
- model architecture registry and embedding/topography assumptions

## C3. Formalization needed for portability

1. Stage interface contract (inputs, outputs, identity fields, artifact schema).
2. Declarative stage manifest model (single source that can derive registry/schema/pipeline wiring).
3. Config contract with explicit required/optional/conditional semantics per stage.
4. Unified defaults policy and validation beyond training-only rules.

## C4. Extensibility blockers to address

1. New stage requires multi-file synchronization by hand (hash registry + telemetry schema + script + pipeline + docs + helper layer).
2. No single declarative source for run-kind definition.
3. Inconsistent config schema authority (YAML runtime truth vs partially stale structured config).

---

## 6) Coherence audit and corrective decisions

## D1. High-priority incoherences

1. **Groups filter key inconsistency**
   - Files under `/home/runner/work/ConTopo/ConTopo/conf/groups/` mix `param.*` and `params.*`.
   - MLflow filter entity keys should be `params.*`, `tags.*`, `attributes.*`.
   - Decision: normalize all group filter keys to MLflow entity conventions.

2. **Structured config staleness**
   - `/home/runner/work/ConTopo/ConTopo/src/config/structured.py` does not fully match active YAML/runtime fields (e.g., some groups/runtime/mlflow additions).
   - Decision: either restore structured config as authoritative and fully synchronize, or explicitly demote its role.

3. **Docs drift on artifacts / setup details**
   - Some docs still mention artifacts or setup paths no longer matching scripts.
   - Decision: choose single canonical artifact inventory + installation flow and align all docs.

## D2. Lower-priority but important incoherences

1. Helper API inconsistencies (mixed dataframe types, vestigial args).
2. Minor naming irregularities between docs and implementation details.
3. Ambiguities around planned optional features (activation maps currently only conceptual).

---

## 7) Implementation order (recommended execution plan)

1. Resolve activation maps scope questions (Section A3) with stakeholder.
2. Lock activation_maps identity + telemetry contract.
3. Implement activation_maps stage + pipeline wiring.
4. Redesign `mlflow_helpers.py` with generic kind API + run_id-first loaders.
5. Update notebooks that depend on old helper wrappers.
6. Fix coherence issues in `conf/groups/*` filter keys.
7. Align docs to code and new stage.
8. Add/adjust tests for:
   - idempotency registry parity
   - telemetry kind coverage
   - new stage skip/recompute behavior

---

## 8) Acceptance criteria for handoff completion

Implementation is considered complete when:

1. `activation_maps` is a first-class run kind with working idempotency + telemetry validation + artifacts.
2. Pipeline can execute/skip activation maps correctly with `execution.force` semantics.
3. `mlflow_helpers.py` exposes a single kind-parameterized list function and simple run_id-based artifact loaders.
4. All docs reflect actual code behavior and new stage.
5. Config/filter inconsistencies are resolved or explicitly documented as intentional.
6. Another agent can run from this plan without needing hidden assumptions.

---

## 9) Explicit unresolved decisions requiring user confirmation

1. Final semantic definition of “activation maps.”
2. Stage location: `03c` standalone vs merge into diagnostics.
3. Required artifact formats and retention expectations.
4. Whether small pipeline includes activation maps by default.
5. DataFrame backend standard in `mlflow_helpers.py` (pandas-only vs mixed support).

