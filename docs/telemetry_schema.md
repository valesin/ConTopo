# Telemetry Schema

**Single Source of Truth:** [src/mlflow_schema_logger.py](src/mlflow_schema_logger.py)

**Rule: any param, tag, metric, or artifact logged to MLflow must be declared in
`TELEMETRY_SCHEMA`.** New entries go into `"optional"` so that existing runs which
pre-date the field still pass validation. Only promote to `"required"` when all
historical runs in the experiment are guaranteed to have the value.

The ConTopo project enforces MLflow run contents with a positive constraint schema at runtime. The enforcement is implemented in `TELEMETRY_SCHEMA` and exercised by the `start_run` context manager and the validation helper in `src/mlflow_schema_logger.py`.

## What the runtime enforces

- At `with start_run(kind, run_name, tags=...)` completion, the system calls a validator that checks:
  - All required parameters (params) for that `kind` are present in `run.data.params`.
  - All required tags for that `kind` are present in `run.data.tags`.
  - All required metrics for that `kind` were logged in `run.data.metrics`.
  - All required artifact templates resolve and exist in the run's artifact tree.

- If any of the above fail, a `TelemetryContractError` is raised. Because this happens inside the MLflow run context, MLflow will mark the run as `FAILED` and the error surface is visible to pipeline control.

## Available helpers

Use the provided helpers in `src/mlflow_schema_logger.py` to log and validate:

- `start_run(kind: str, run_name: str, tags: Mapping[str, Any] | None = None)` — context manager that starts an MLflow run, enforces allowed tags at start, and validates the full telemetry contract when the block completes successfully.
- `log_params(kind: str, params: Mapping[str, Any])` — logs params after checking they are allowed for the `kind`.
- `log_tags(kind: str, tags: Mapping[str, Any])` — sets tags after checking they are allowed for the `kind`.
- `timed_log_metric(s)`, `timed_log_artifact`, `timed_log_model` — convenience wrappers that add timing output when logging.

Note: the `start_run` call performs an early check that the tags you provide are within the allowed set (required + optional). Attempting to set an unknown tag at start will raise a `ValueError`.

## Artifact path resolution

Artifact templates in `TELEMETRY_SCHEMA` can include `{...}` placeholders that are resolved against the run's tags and params at validation time. The implementation merges tags and params into a single formatting context (`{**run.data.tags, **run.data.params}`) with `params` values overriding `tags` when keys collide.

If a template cannot be formatted because a parameter/tag is missing, validation raises `TelemetryContractError` with a clear message naming the missing key.

Artifact existence is checked with the MLflow tracking client's `list_artifacts()` for the artifact's directory; a match is accepted if an artifact item's `path` equals the expected path or starts with `expected_path + "/"` (to allow folders).

## Run kinds

The currently defined run kinds (keys in `TELEMETRY_SCHEMA`) are:

- `model`
- `inference`
- `category_similarity_profile`
- `diagnostics`
- `ensemble`
- `diversity`
- `consistency`
- `metalearner`

Each kind defines four slots: `params`, `tags`, `metrics`, and `artifacts`. For each slot the schema lists `required` and `optional` entries. The validator only enforces presence of `required` entries at run completion; the helper functions enforce that any param/tag you attempt to log is within the allowed (required + optional) set.

## How to update the schema safely

1. Edit `TELEMETRY_SCHEMA` in [src/mlflow_schema_logger.py](src/mlflow_schema_logger.py): add required/optional names or artifact templates under the appropriate `kind` entry.
2. Update the runtime code that logs these values (scripts or modules): use `log_params(kind, {...})`, `log_tags(kind, {...})`, `timed_log_metric` / `timed_log_metrics`, or `timed_log_artifact` / `timed_log_model` to ensure values are emitted using the canonical names.
3. Run a quick validation run. Example pattern (Hydra overrides shown as an example):

```bash
python -m main pipeline.from_step=some_step pipeline=small
```

Successful validation prints the `PASS` message from `start_run`:

```
[VALIDATION] Enforcing telemetry contract for <run_name> (kind=<kind>)... PASS
```

## Failure modes to watch for

- Missing required params/tags/metrics will raise `TelemetryContractError` naming the missing keys.
- Artifact template resolution failures raise `TelemetryContractError` identifying the missing formatting key.
- If the artifact exists but under a different path than the template resolves to, the validator will report the expected path as missing.
- Attempting to log unknown param/tag keys via `log_params`/`log_tags` will raise a `ValueError` (these functions proactively enforce allowed keys).

### Worked example — missing required metric

Suppose you add `test_accuracy_topk=5` to the `model` kind's required metrics
but forget to log it from `scripts/01_train_models.py`. At run completion:

```
[VALIDATION] Enforcing telemetry contract for resnet18_rho0.05_torus_t0 (kind=model)... FAIL
TelemetryContractError: run <run_id> kind=model missing required metrics:
  - test_accuracy_topk
```

The MLflow run is marked `FAILED`. Fix:

1. Add the `timed_log_metric("test_accuracy_topk", value)` call at the
   appropriate point in the script (typically near the existing `test_accuracy`
   log site).
2. Re-run. Successful validation prints `PASS`.

### Worked example — unknown tag at `start_run`

Passing a tag not declared in `TELEMETRY_SCHEMA[kind]["tags"]` (either list):

```python
with start_run("model", run_name="...", tags={"my_custom_tag": "x"}):
    ...
```

fails immediately, before any work begins:

```
ValueError: kind=model disallowed tags: ['my_custom_tag']
Allowed (required + optional): ['cfg_hash', 'identity_hash', 'parent_run_id', ...]
```

Fix: either add `my_custom_tag` to the kind's `"optional"` tag list in
`TELEMETRY_SCHEMA`, or (preferably) use an existing declared tag that captures
the same semantic.

## Notes and best practices

- Prefer modifying `TELEMETRY_SCHEMA` first, then the code that emits the telemetry. That order prevents unexpected validation failures during CI or local runs.
- When adding artifact templates, prefer stable directory prefixes (e.g., `inference/`, `ensemble/`, `profiles/`) so the validator's directory listing check is robust.
- If a new telemetry item changes step identity semantics, update `src/config/hash.py`'s `IDEMPOTENCY_REGISTRY` and add tests to cover idempotency.
- When adding new `TrainingConfig` fields, add them to the `"optional"` list in `TELEMETRY_SCHEMA["model"]["params"]` so that existing FINISHED runs (which pre-date the field) still pass validation. Do **not** add them to `"required"` unless all historical runs are guaranteed to have them.

## Currently optional `model` params (added after initial schema)

The following `model`-kind params are declared `optional` because they were added
after runs already existed in the experiment:

- `loading_backend` — `"torch"` or `"ffcv"` (hash-included; logged for observability)
- `label_smoothing` — CrossEntropyLoss smoothing coefficient
- `use_blurpool` — antialiased pooling flag
- `optimizer_selective_wd` — selective weight decay flag
- `lr_tta` — test-time augmentation flag
- `lr_peak_epoch` — OneCycleLR peak epoch (only set when `scheduler=cyclic`; `None` otherwise)
- `progressive_res_min` — progressive resolution start size
- `progressive_res_max` — progressive resolution end size
- `progressive_res_start_ramp` — ramp start fraction (only set when progressive res is active; `None` otherwise)
- `progressive_res_end_ramp` — ramp end fraction (only set when progressive res is active; `None` otherwise)
- `beton_max_resolution` — beton max image dimension (only set when `loading_backend=ffcv`; `None` otherwise)
- `beton_jpeg_quality` — beton JPEG quality (only set when `loading_backend=ffcv`; `None` otherwise)
- `beton_compress_probability` — beton JPEG fraction (only set when `loading_backend=ffcv`; `None` otherwise)

## Reference

Implementation: [src/mlflow_schema_logger.py](src/mlflow_schema_logger.py)
