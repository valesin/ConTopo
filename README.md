# ConTopo

ConTopo is a Hydra + PyTorch + MLflow research pipeline for topographic regularization experiments on CIFAR-10 and ImageNet100.

Pipeline stages:
1. Train base models.
2. Cache inference artifacts.
3. Compute profile and diagnostic artifacts.
4. Build ensembles and post-ensemble analyses.
5. Train meta-learners (adapters).

## Quick start

### 1) Environment

```bash
uv python pin 3.13
uv sync
```

Optional analysis extras:

```bash
uv sync --group analysis
```

### 2) Secrets and tracking configuration

Copy and load secrets:

```bash
cp .env.secrets.example .env.secrets
chmod 600 .env.secrets
# edit .env.secrets
source .env.secrets
```

By default, scripts use local tracking (`outputs/mlflow.db` and `outputs/mlruns`).
To use remote tracking/storage, pass Hydra overrides with conditional expansion so unset variables are skipped safely:

```bash
${MLFLOW_TRACKING_URI:+mlflow.tracking_uri="$MLFLOW_TRACKING_URI"}
${MLFLOW_ARTIFACT_LOCATION:+mlflow.artifact_location="$MLFLOW_ARTIFACT_LOCATION"}
${MLFLOW_EXPERIMENT_NAME:+mlflow.experiment_name="$MLFLOW_EXPERIMENT_NAME"}
```

### 3) Run the pipeline

Full pipeline:

```bash
python main.py \
	${MLFLOW_TRACKING_URI:+mlflow.tracking_uri="$MLFLOW_TRACKING_URI"} \
	${MLFLOW_ARTIFACT_LOCATION:+mlflow.artifact_location="$MLFLOW_ARTIFACT_LOCATION"} \
	${MLFLOW_EXPERIMENT_NAME:+mlflow.experiment_name="$MLFLOW_EXPERIMENT_NAME"}
```

Smoke preset:

```bash
python main.py pipeline=small
```

Resume from a step:

```bash
python main.py pipeline.from_step=ensemble
```

## Pipeline scripts

- `scripts/01_train_models.py`: train one model run (`kind=model`).
- `scripts/02_cache_inference.py`: cache logits/probs/embeddings (`kind=inference`).
- `scripts/03_compute_profiles.py`: category-similarity profiles (`kind=category_similarity_profile`).
- `scripts/03b_compute_diagnostics.py`: optional per-model diagnostics (`kind=diagnostics`).
- `scripts/04_run_ensemble.py`: vote-based ensembles (`kind=ensemble`).
- `scripts/04b_compute_diversity.py`: diversity metrics (`kind=diversity`).
- `scripts/04c_compute_consistency.py`: RDM/RSA consistency (`kind=consistency`).
- `scripts/05_train_adapters.py`: meta-learner training (`kind=metalearner`).

The orchestrator reads `conf/pipeline/default.yaml` (or `conf/pipeline/small.yaml`).

### Per-step overrides

Each step in the pipeline YAML can declare a `sweep` and an `overrides` list.
Values in `overrides` are forwarded as Hydra CLI overrides when the orchestrator
launches that step as a subprocess:

```yaml
- id: inference
  script: 02_cache_inference.py
  sweep: training_rho_loss
  overrides:
    - "loss.rho=0.05"
    - "trial=0"
```

`main.py` runs that step roughly as:

```bash
python scripts/02_cache_inference.py +sweeps=training_rho_loss loss.rho=0.05 trial=0
```

Use this to pin per-step params without editing the top-level CLI. See
`conf/pipeline/` for working examples.

## Docker runs

Build image:

```bash
docker build -t ghcr.io/${SKYPILOT_DOCKER_USERNAME}/contopo:latest .
```

If the image is private, login first:

```bash
echo "$SKYPILOT_DOCKER_PASSWORD" | docker login ghcr.io -u "$SKYPILOT_DOCKER_USERNAME" --password-stdin
```

Run a training command in Docker:

```bash
docker run --gpus all \
	-e MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
	-e MLFLOW_ARTIFACT_LOCATION="$MLFLOW_ARTIFACT_LOCATION" \
	-e MLFLOW_EXPERIMENT_NAME="$MLFLOW_EXPERIMENT_NAME" \
	-e MLFLOW_S3_ENDPOINT_URL="$MLFLOW_S3_ENDPOINT_URL" \
	-e MLFLOW_TRACKING_USERNAME="$MLFLOW_TRACKING_USERNAME" \
	-e MLFLOW_TRACKING_PASSWORD="$MLFLOW_TRACKING_PASSWORD" \
	-e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
	-e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
	ghcr.io/${SKYPILOT_DOCKER_USERNAME}/contopo:latest \
	python scripts/01_train_models.py loss.rho=0.05 trial=0 \
		${MLFLOW_TRACKING_URI:+mlflow.tracking_uri="$MLFLOW_TRACKING_URI"} \
		${MLFLOW_ARTIFACT_LOCATION:+mlflow.artifact_location="$MLFLOW_ARTIFACT_LOCATION"} \
		${MLFLOW_EXPERIMENT_NAME:+mlflow.experiment_name="$MLFLOW_EXPERIMENT_NAME"}
```

## Cloud runs (SkyPilot)

Single job:

```bash
sky jobs launch sky_task.yaml \
	--env LOSS_RHO=0.0 --env TRIAL=0 \
	--env MLFLOW_TRACKING_URI --env MLFLOW_ARTIFACT_LOCATION --env MLFLOW_EXPERIMENT_NAME \
	--env MLFLOW_S3_ENDPOINT_URL --env MLFLOW_TRACKING_USERNAME \
	--secret AWS_ACCESS_KEY_ID --secret AWS_SECRET_ACCESS_KEY \
	--secret MLFLOW_TRACKING_PASSWORD \
	--secret SKYPILOT_DOCKER_USERNAME --secret SKYPILOT_DOCKER_SERVER --secret SKYPILOT_DOCKER_PASSWORD
```

Full sweep launcher:

```bash
source .env.secrets && ./launch_sweep.sh
```

Monitor:

```bash
sky jobs queue
sky jobs logs <job_name>
```

## Dataset notes

The pipeline is dataset-agnostic. The active dataset is controlled by the `dataset`
config group (`conf/dataset/<name>.yaml`). The default is CIFAR-10.

Use a distinct `mlflow.experiment_name` per dataset to prevent ensemble discovery
from mixing models trained on different datasets. The ImageNet100 sweep sets
`mlflow.experiment_name=contopo_imagenet100` automatically.

### Built-in datasets

| Config | Dataset | Classes | Image size | Loader |
|---|---|---|---|---|
| `cifar10` | CIFAR-10 | 10 | 32×32 | torchvision auto-download |
| `imagenet100` | ImageNet100 | 100 | 224×224 | ImageFolder at `<data_root>/imagenet100/` |

### Split policy

All datasets use `first_n_per_class` by default: the first `val_per_class` images
per class (by dataset ordering) are reserved for validation; the rest form the
training set. The test split is the dataset's native held-out partition.

| Dataset | `val_per_class` | Val size | Train size |
|---|---|---|---|
| CIFAR-10 | 500 | 5 000 | 45 000 |
| ImageNet100 | 50 | 5 000 | 45 000 |

### Adding a new dataset

See [`docs/contributing.md`](docs/contributing.md) §10 for the step-by-step guide.

## Training backends

`scripts/01_train_models.py` supports two data loading backends, selected via
`training.loading_backend` (default: `torch`).

### `torch` (default)

Standard `torch.utils.data.DataLoader`. Works for all datasets, no additional
dependencies. All existing CIFAR-10 sweep configs use this path.

### `ffcv`

High-throughput binary data loading for large-image datasets (e.g. ImageNet100).
Requires the optional `ffcv` dependency group:

```bash
uv sync --group ffcv
```

This installs `ffcv`, `antialiased-cnns` (blurpool), and `cupy-cuda12x` (GPU-side
normalisation). Adjust the cupy package name if your CUDA version differs from 12.x.

`.beton` data files are **generated automatically on first run** and reused
thereafter. The full FFCV training recipe includes:

| Feature | Config key | FFCV sweep default |
|---|---|---|
| SGD optimiser | `training.optimiser: sgd` | ✓ |
| OneCycleLR scheduler | `training.scheduler: cyclic` | ✓ |
| LR peak epoch | `training.lr_peak_epoch: 2` | ✓ (required for cyclic) |
| Label smoothing | `training.label_smoothing: 0.1` | ✓ |
| Blurpool | `training.use_blurpool: true` | ✓ |
| Selective weight decay | `training.optimizer_selective_wd: true` | ✓ |
| Test-time augmentation | `training.lr_tta: true` | ✓ (ffcv only) |
| Progressive resolution | `training.progressive_res_min/max: 160/192` | ✓ (ffcv only) |
| Progressive ramp | `training.progressive_res_start_ramp/end_ramp: 0.75/1.0` | ✓ (required when prog res active) |

Use the FFCV sweep preset for a full ImageNet100 FFCV training run:

```bash
uv run scripts/01_train_models.py +sweeps=training_rho_imagenet100_ffcv \
  loss.rho=0.0 loss.topology=torus trial=0
```

> **Migration note**: If you have existing model runs, run the migration scripts
> before deploying this change. See `docs/ffcv_param_assumptions.md` for the
> full guide and commands.

### Config validation

The training script validates the composed config at startup and fails fast on
invalid combinations (e.g. `scheduler=cyclic` without `lr_peak_epoch`). See
[`docs/config_system.md`](docs/config_system.md) §5 for the full rule set.

## Running experiments under different conditions

The pipeline is configuration-driven — most changes are CLI overrides, no
code edits needed. Common operational patterns:

### Use a specific dataset

```bash
# CIFAR-10 (default)
python main.py

# ImageNet100 + ResNet34
python main.py dataset=imagenet100 model=resnet34_imagenet100 \
  profiling=imagenet100 mlflow.experiment_name=contopo_imagenet100
```

### Switch the training backend

```bash
# torch DataLoader (default; works everywhere)
python scripts/01_train_models.py training.loading_backend=torch ...

# FFCV (fast binary loading — requires `uv sync --group ffcv`)
python main.py +sweeps=training_rho_imagenet100_ffcv
```

See [Training backends](#training-backends) above for the full FFCV recipe.

### Smoke-test with a single epoch

Use `training.epochs=1` for any verification run; never launch a full training
by accident when testing.

```bash
python scripts/01_train_models.py training.epochs=1 trial=99
```

### Resume from a specific pipeline step

```bash
python main.py pipeline.from_step=ensemble
```

### Force re-execution of a stage

Every post-training stage checks idempotency and skips if a matching
`FINISHED` run already exists. To force re-execution:

```bash
python scripts/02_cache_inference.py execution.force=true loss.rho=0.0 trial=0
```

### Change the ensemble group definition

Steps 04, 04b, 04c, and 05 use `cfg.groups` to discover which finished model
runs to combine into ensembles. The active group config is selected with
`groups=<name>` (default: `default`).

| Config | `group_by` | `sample_size` | `filter` | Effect |
|---|---|---|---|---|
| `default` | `[topology, rho]` | `null` (full group) | none | One ensemble per `(topology, rho)` pair using all component runs |
| `samples9` | `[topology, rho]` | `2` (all pairs) | `{params.epochs: "1"}` | All C(n, 2) pairs per group; scoped to 1-epoch runs |

Filter keys use **full MLflow entity paths** (`params.<name>`, `tags.<name>`,
`attributes.<name>`), not Hydra config paths.

```bash
# Run ensemble step with default grouping (one ensemble per rho/topology pair)
python scripts/04_run_ensemble.py

# Run ensemble step with k-combination sampling (all pairs within each group)
python scripts/04_run_ensemble.py groups=samples9

# Re-run all downstream steps from ensemble onward with a different grouping
python main.py pipeline.from_step=ensemble groups=samples9

# Ad-hoc: filter to a specific topology without creating a new config
python scripts/04_run_ensemble.py "groups.filter={params.topology: torus}"

# Ad-hoc: filter by a tag (e.g. specific trial)
python scripts/04_run_ensemble.py "groups.filter={tags.trial: '3'}"

# Ad-hoc: change k-combination size on the fly
python scripts/04_run_ensemble.py groups.sample_size=3
```

Steps 04b, 04c, and 05 accept the same `groups=` override. Use a consistent
group definition across all downstream steps so ensemble identity hashes match.

### Use a different MLflow experiment

```bash
python main.py mlflow.experiment_name=my_experiment_name ...
```

The active dataset config automatically sets its own experiment name
(e.g. `contopo_imagenet100` for the ImageNet100 sweep) to keep ensembles
from mixing models across datasets.

### Run a custom sweep

Define a YAML under `conf/sweeps/<name>.yaml` and activate it with
`+sweeps=<name>`:

```bash
python main.py +sweeps=training_small_ffcv_cifar
```

## Configuration model

All Hydra config groups live under `conf/`. For the full config reference
(groups, fields, hash-included vs. hash-excluded, validation rules) see
[`docs/config_system.md`](docs/config_system.md).

## Minimal command set

Run one model config:

```bash
python scripts/01_train_models.py loss.rho=0.05 loss.topology=grid trial=0
```

Cache inference for same config:

```bash
python scripts/02_cache_inference.py loss.rho=0.05 loss.topology=grid trial=0
```

Compute profiles and diagnostics:

```bash
python scripts/03_compute_profiles.py loss.rho=0.05 loss.topology=grid trial=0
python scripts/03b_compute_diagnostics.py loss.rho=0.05 loss.topology=grid trial=0
```

Run downstream singleton stages once per experiment:

```bash
# Group-based steps — discover models from MLflow, no sweep params needed
python scripts/04_run_ensemble.py
python scripts/04b_compute_diversity.py
python scripts/04c_compute_consistency.py
python scripts/05_train_adapters.py +sweeps=metalearning
```

Steps 01–03b are **sweep-based**: they accept per-model Hydra params (e.g.
`loss.rho=0.05 trial=0`) and run one MLflow run per config.
Steps 04–05 are **group-based**: they run once per experiment and discover
which trained models to combine via `conf/groups/`.

## Developer references

Deep internals and extension guides live in `docs/`:

- [`docs/README.md`](docs/README.md) — index and where-to-start
- [`docs/architecture.md`](docs/architecture.md) — runtime architecture, stage flow, MLflow boundaries
- [`docs/config_system.md`](docs/config_system.md) — Hydra groups, hash inclusion, validation rules, adding parameters
- [`docs/idempotency.md`](docs/idempotency.md) — identity hashes, registry, migration semantics
- [`docs/telemetry_schema.md`](docs/telemetry_schema.md) — MLflow logging contract per run kind
- [`docs/contributing.md`](docs/contributing.md) — safe change procedures and migration checklists
- [`docs/ffcv_param_assumptions.md`](docs/ffcv_param_assumptions.md) — worked example: FFCV migration
- [`docs/analysis_guide.md`](docs/analysis_guide.md) — notebook and MLflow analysis reference
