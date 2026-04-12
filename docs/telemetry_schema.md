# Telemetry Logging & Schema Contracts

**Single Source of Truth:** [src/mlflow_schema_logger.py](src/mlflow_schema_logger.py)

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
python -m main pipeline.from_step=some_step pipeline.to_step=some_step +pipeline=small
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

## Notes and best practices

- Prefer modifying `TELEMETRY_SCHEMA` first, then the code that emits the telemetry. That order prevents unexpected validation failures during CI or local runs.
- When adding artifact templates, prefer stable directory prefixes (e.g., `inference/`, `ensemble/`, `profiles/`) so the validator's directory listing check is robust.
- If a new telemetry item changes step identity semantics, update `src/config/hash.py`'s `IDEMPOTENCY_REGISTRY` and add tests to cover idempotency.

## Reference

Implementation: [src/mlflow_schema_logger.py](src/mlflow_schema_logger.py)
