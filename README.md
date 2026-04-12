# ConTopo

ConTopo is a Hydra + PyTorch + MLflow research pipeline for topographic regularization experiments on CIFAR-10.

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

Passing additional overrides from the pipeline definition

The pipeline step graph (`conf/pipeline/*.yaml`) may include per-step overrides.
Each step can declare a `sweep` (becomes `+sweeps=<name>`) and an `overrides` list.
Values in `overrides` are forwarded as Hydra CLI overrides to the child script when
the orchestrator launches it as a subprocess. Example step excerpt:

```yaml
	- id: inference
		script: 02_cache_inference.py
		sweep: training_rho_loss
		overrides:
			- "loss.rho=0.05"
			- "trial=0"
```

When `main.py` runs that step, the script is executed roughly as:

```bash
python scripts/02_cache_inference.py +sweeps=training_rho_loss loss.rho=0.05 trial=0
```

Use this to pin per-step params (for example, a specific rho or trial) without
editing the top-level CLI. This is useful for preset pipelines in `conf/pipeline`.

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

- Dataset: CIFAR-10 (`conf/dataset/cifar10.yaml`).
- Split policy: `first_n_per_class`.
- Default `val_per_class=500` gives 5,000 validation and 45,000 training samples.
- The split is deterministic by CIFAR-10 original order (not random per run).

## Configuration model

Main config: `conf/config.yaml`

Primary groups:

- `model`, `loss`, `dataset`, `training`: model-identity inputs.
- `runtime`, `mlflow`: execution and tracking environment.
- `execution`: split + force controls.
- `profiling`: anchors, profile metrics, diagnostics toggles.
- `groups`: component discovery for ensemble/analysis stages.
- `ensemble`: vote methods.
- `adapter`: metalearner setup.
- `pipeline`: orchestrator step graph.
- `sweeps`: multirun presets.

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
python scripts/04_run_ensemble.py
python scripts/04b_compute_diversity.py
python scripts/04c_compute_consistency.py
python scripts/05_train_adapters.py
```

## Developer references

- `ARCHITECTURE.md`: runtime architecture and boundaries.
- `CONTRIBUTING_AND_UPDATING.md`: safe change procedures.
- `ANALYSIS_GUIDE.md`: notebook/query workflows.
- `doc_drift_report.md`: documentation drift findings and resolution map.
