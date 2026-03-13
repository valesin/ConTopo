# ConTopo — Exhaustive Codebase Analysis

> **Generated:** 2026-03-13  
> **Scope:** Every file in the repository — source code, configuration, tests, and scripts.  
> **Purpose:** A complete reference for anyone entering this codebase: structure, rationale, logic flows, identified issues, and improvement recommendations.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Configuration System](#3-configuration-system)
4. [Core Library (`src/`)](#4-core-library-src)
   - 4.1 [Data Layer](#41-data-layer)
   - 4.2 [Network Architectures](#42-network-architectures)
   - 4.3 [Loss Functions](#43-loss-functions)
   - 4.4 [Training Loop](#44-training-loop)
   - 4.5 [Inference & Caching](#45-inference--caching)
   - 4.6 [Profiling & Metrics](#46-profiling--metrics)
   - 4.7 [Ensemble Logic](#47-ensemble-logic)
   - 4.8 [MLflow Utilities](#48-mlflow-utilities)
   - 4.9 [Config Utilities](#49-config-utilities)
5. [Pipeline Scripts](#5-pipeline-scripts)
6. [Test Suite](#6-test-suite)
7. [Identified Bugs & Fixes Applied](#7-identified-bugs--fixes-applied)
8. [Design Observations & Potential Improvements](#8-design-observations--potential-improvements)
9. [Security Considerations](#9-security-considerations)
10. [Dependency Analysis](#10-dependency-analysis)
11. [Summary](#11-summary)

---

## 1. Project Overview

**ConTopo** is a research framework for studying **topographic regularisation** in neural networks, specifically investigating how enforcing spatial structure on embedding layers affects model diversity and ensemble performance.

### Core research questions

| Question | How ConTopo addresses it |
|----------|--------------------------|
| Does topographic loss improve single-model accuracy? | Step 01: trains models with varying `rho` (topographic weight) |
| How does it affect representational structure? | Steps 03/03b: category-similarity profiles, Moran's I, weight-norm analysis |
| Does topographic diversity improve ensembles? | Steps 04/04b/04c: ensemble voting, diversity metrics, RSA consistency |
| Can meta-learners exploit topographic structure? | Step 05: adapter training on embeddings + similarity profiles |

### Pipeline overview

```
01_train_models.py       → Train CIFAR-10 classifiers (ResNet18 + topographic loss)
02_cache_inference.py    → Cache logits, embeddings, predictions per model
03_compute_profiles.py   → Compute per-sample similarity profiles vs anchor embeddings
03b_compute_diagnostics.py → Moran's I, weight norms, unit-distance correlation
04_run_ensemble.py       → Ensemble voting (soft, hard, max-confidence, conf-weighted)
04b_compute_diversity.py → Pairwise diversity metrics (Q-statistic, disagreement, etc.)
04c_compute_consistency.py → RDM/RSA consistency across ensemble components
05_train_adapters.py     → Meta-learner training (linear / MLP) on ensemble features
```

Each step is tracked via **MLflow** runs with **idempotency** — existing results are detected by content hashes and skipped unless `pipeline.force=true`.

---

## 2. Repository Structure

```
ConTopo/
├── conf/                          # Hydra YAML config groups
│   ├── config.yaml                # Master composition (defaults list)
│   ├── adapter/default.yaml       # Meta-learner training config
│   ├── dataset/cifar10.yaml       # Dataset specification
│   ├── ensemble/                  # Ensemble definitions
│   │   ├── ce_ensembles.yaml      # Default: 6 rho × 2 topologies × 5 trials
│   │   ├── full_sweep.yaml        # Production: 5 rho × 2 × 10 trials
│   │   └── small_grid.yaml        # CI/debug: 2 rho × 2 trials
│   ├── loss/cross_entropy.yaml    # Loss config (rho, topology, neighborhood)
│   ├── mlflow/default.yaml        # MLflow tracking config
│   ├── model/resnet18.yaml        # Model architecture config
│   ├── pipeline/default.yaml      # Pipeline flow control
│   ├── runtime/default.yaml       # Device, paths, workers
│   └── training/default.yaml      # Training hyperparameters
├── scripts/                       # Pipeline entry points (numbered)
│   ├── 01_train_models.py
│   ├── 02_cache_inference.py
│   ├── 03_compute_profiles.py
│   ├── 03b_compute_diagnostics.py
│   ├── 04_run_ensemble.py
│   ├── 04b_compute_diversity.py
│   ├── 04c_compute_consistency.py
│   └── 05_train_adapters.py
├── src/                           # Core library
│   ├── config/                    # Configuration utilities
│   │   ├── hash.py                # Deterministic config hashing
│   │   ├── paths.py               # Centralized path resolution
│   │   ├── schema.py              # Schema versioning
│   │   └── structured.py          # Hydra dataclass schemas
│   ├── data/                      # Data management
│   │   ├── anchors.py             # Anchor selection from manifests
│   │   ├── cache.py               # Storage backend abstraction
│   │   ├── loaders.py             # CIFAR-10 DataLoaders
│   │   ├── manifest.py            # Dataset content manifests
│   │   └── transforms.py          # Named transform presets
│   ├── ensemble/                  # Ensemble combination
│   │   ├── accuracy.py            # Ensemble accuracy utilities
│   │   ├── combine.py             # Logit combination methods
│   │   └── selector.py            # Declarative component selector
│   ├── losses/                    # Loss functions
│   │   ├── balancer.py            # Gradient-norm loss balancer
│   │   └── topographic.py         # Topographic regularization losses
│   ├── networks/                  # Model definitions
│   │   ├── __init__.py            # Re-exports
│   │   ├── heads.py               # Adapter heads (Linear, MLP)
│   │   ├── registry.py            # Model factory
│   │   └── resnet18.py            # Modified ResNet18 for CIFAR-10
│   ├── profiling/                 # Analysis & metrics
│   │   ├── category_similarity.py # Per-sample similarity profiles
│   │   ├── diversity.py           # Pairwise diversity metrics
│   │   ├── rdm.py                 # RDM / RSA computations
│   │   ├── smoothness.py          # Moran's I spatial autocorrelation
│   │   └── unit_analysis.py       # Weight-based unit analysis
│   ├── training/                  # Training utilities
│   │   ├── checkpoint.py          # Save/load checkpoints
│   │   └── train_ce.py            # Cross-entropy training loop
│   ├── __init__.py
│   ├── inference.py               # Inference runner + caching
│   └── mlflow_utils.py            # MLflow helpers (tags, idempotency)
├── tests/                         # Test suite (128 tests)
│   ├── test_anchor_determinism.py
│   ├── test_cache_alignment.py
│   ├── test_category_similarity.py
│   ├── test_cfg_hash.py
│   ├── test_ensemble.py
│   ├── test_hydra_config.py
│   └── test_profile_gating.py
├── pyproject.toml                 # Project metadata & dependencies
└── README.md
```

---

## 3. Configuration System

### 3.1 Hydra Composition

The project uses **Hydra** with **structured configs** (Python dataclasses) for schema validation.

**Master config** (`conf/config.yaml`) composes groups via a defaults list:
```yaml
defaults:
  - model: resnet18
  - loss: cross_entropy
  - dataset: cifar10
  - training: default
  - runtime: default
  - pipeline: default
  - mlflow: default
  - ensemble: ce_ensembles
  - adapter: default
  - _self_
```

Each group maps to a dataclass in `src/config/structured.py`, registered via `register_configs()`.

### 3.2 Config Hashing (`src/config/hash.py`)

Deterministic SHA-256 hash (16 hex chars) of experiment-semantic config only.

**Excluded keys** (do not affect experiment results):
```python
EXCLUDED_KEYS = frozenset({
    "runtime", "mlflow", "storage", "hydra",
    "pipeline", "ensemble", "adapter", "migration"
})
```

**Included** (experiment-semantic): `schema_version`, `trial`, `seed`, `model.*`, `loss.*`, `dataset.*`, `training.*`

Process: resolve OmegaConf interpolations → strip excluded keys → deep-sort → `json.dumps(sort_keys=True)` → SHA-256[:16].

### 3.3 Schema Versioning (`src/config/schema.py`)

`SCHEMA_VERSION = 1` — bumped when config field meanings change. `apply_schema_defaults()` ensures required fields exist and auto-derives seed from trial (`seed = 100 + trial`).

### 3.4 Path Resolution (`src/config/paths.py`)

All output paths are derived from `cfg.runtime.outputs_root`:
- `get_models_dir()` → `outputs/models`
- `get_cache_dir()` → `outputs/cache`
- `get_analysis_dir()` → `outputs/analysis`
- `get_mlflow_db_path()` → parses `sqlite:///outputs/mlflow.db`

`ensure_output_dirs()` creates all directories before any MLflow operations.

### 3.5 Key Config Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `loss.rho` | `0.0` | Topographic regularization weight (sweep variable) |
| `loss.topology` | `torus` | Grid topology (`grid` or `torus`) |
| `loss.topography_type` | `ws` | Loss type (`ws` = weight smoothing, `global`) |
| `model.embedding_dim` | `256` | Embedding dimension (determines grid shape) |
| `training.epochs` | `200` | Maximum training epochs |
| `training.early_stopping_patience` | `25` | Epochs without improvement before stopping |
| `adapter.feature_type` | `logits` | Meta-learner input features |
| `adapter.similarity_metric` | `cosine` | Similarity metric for profiles |

---

## 4. Core Library (`src/`)

### 4.1 Data Layer

#### `src/data/manifest.py` — Dataset Manifests

**Purpose:** Stable alignment across runs via content-hashed example IDs.

**`DatasetManifest`** dataclass contains:
- `example_ids` — SHA-256[:16] of raw PIL image bytes (content-hash, not index-based)
- `original_indices` — position in canonical dataset ordering
- `labels` — ground-truth class labels
- `manifest_hash` — deterministic hash of `"|".join(example_ids) + "|" + split`

**`get_or_create_manifest()`** — Idempotent creation:
- Caches at `<artifacts_root>/dataset_manifests/<dataset>/<split>/manifest.pt`
- For `val` split: selects first `val_per_class` examples per class from train set
- For `train` split: excludes val indices from full train set
- For `test` split: builds directly from test set

**Design note:** The manifest is model-independent — it depends only on dataset content, ensuring different training runs align on the same examples.

#### `src/data/anchors.py` — Anchor Selection

**Purpose:** Deterministic selection of anchor (reference) examples for similarity profiles.

**`AnchorSpec`** — frozen dataclass: `source_split`, `per_class`, `strategy`, `order_by`, `num_classes`. All fields required; defaults live in Hydra structured configs.

**`select_anchors_from_manifest()`** — Strategy-based selection:
- Currently only `per_class_first_n` strategy implemented
- Sorts each class by `order_by` (`example_id` or `original_index`)
- Selects first `per_class` per class
- Returns dict with `anchor_indices`, `anchor_example_ids`, `anchor_labels`, `spec`, `spec_hash`

**`get_or_create_anchors()`** — Caches at `<artifacts_root>/anchors/<dataset>/<split>/<spec_hash>/anchors.pt`

#### `src/data/loaders.py` — DataLoaders

**Purpose:** CIFAR-10 DataLoaders with deterministic train/val split.

**`_split_train_val_indices()`** — Deterministic 45k/5k split: takes first `val_per_class` examples per class as validation. Compatible with manifest val logic.

**`get_cifar10_loaders()`** — Returns `(train_loader, val_loader, test_loader)` from Hydra config.

**`get_cifar10_eval_loader()`** — Standalone test loader for inference (no train augmentation).

#### `src/data/transforms.py` — Transform Presets

Named, versioned `(train_transform, eval_transform)` pairs:
- `cifar10_default_v1` — RandomCrop(32, padding=4) + HFlip
- `cifar10_resizedcrop_v1` — RandomResizedCrop(32, scale=(0.2, 1.0)) + HFlip

Both normalize with CIFAR-10 mean/std. Preset names are included in `cfg_hash`, so different augmentations produce different hashes.

#### `src/data/cache.py` — Storage Backends

Abstract `StorageBackend` with two implementations:
- **`PtBackend`** — `.pt` files via `torch.save`/`torch.load`
- **`ZarrBackend`** — Stub (`NotImplementedError`) for future use

**Note:** `PtBackend.load()` uses `weights_only=False` for backward compatibility with complex saved objects.

### 4.2 Network Architectures

#### `src/networks/resnet18.py` — Modified ResNet18

**`Block`** — Standard residual block with shortcut projection when dimensions change.

**`ResNet18`** — Modified for CIFAR-10:
- Stride-1 first conv (vs. stride-2 in standard ResNet) — preserves 32×32 spatial resolution
- Layer progression: 64 → 128 → 256 → 512
- AdaptiveAvgPool + Linear(512, emb_dim)
- Kaiming initialization for Conv2d, constant init for BatchNorm

**`LinearResNet18`** — Encoder + dropout + linear classifier:
- Returns `(embeddings, logits)` when `ret_emb=True`, otherwise just `logits`
- Supports optional dropout and head bias control

#### `src/networks/heads.py` — Adapter Heads

**`LinearAdapter`** — Single linear layer (`emb_dim → num_classes`), optional bias.

**`TwoLayerMLPAdapter`** — Two-layer MLP: `Linear(in_dim, hidden_dim) → ReLU → Dropout → Linear(hidden_dim, num_classes)`. Optional bias on final layer.

> **Bug fixed:** This class was previously named `ThreeLayerMLPAdapter` despite having only two linear layers. See [Section 7](#7-identified-bugs--fixes-applied).

#### `src/networks/registry.py` — Model Factory

Registry pattern mapping architecture names to classes:
```python
_MODEL_REGISTRY = {"LinearResNet18": LinearResNet18}
```

`build_model(cfg, ret_emb)` instantiates from Hydra config. `unwrap()` strips DataParallel wrapper. `to_device()` moves to device with optional DataParallel wrapping.

### 4.3 Loss Functions

#### `src/losses/topographic.py` — Topographic Losses

**Grid geometry utilities:**
- `get_grid_shape(emb_dim)` — Factors `emb_dim` into `(h, w)` closest to square (e.g., 256 → 16×16)
- `pos_dist(h, w)` — Pairwise Euclidean distance matrix for all grid positions
- `grid_diffs(x, h, w)` — Adjacent-unit differences (no wrapping)
- `torus_diffs(x, h, w)` — Adjacent-unit differences with periodic boundary conditions

**`Global_Topographic_Loss`:**
- Penalizes mismatch between feature correlation and grid distance
- Computes `D_norm = D / D.max()` (normalized distance matrix)
- Loss = MSE between feature correlation matrix and inverted distance matrix
- Operates on **pre-ReLU activations** (before the final linear layer)

**`Local_WS_Loss` (Weight Smoothing):**
- Penalizes differences between weights of adjacent units on the topographic grid
- Supports `grid` and `torus` topologies
- Loss = mean of squared weight differences across all adjacent pairs

**Design note:** The torus diagonal wrapping includes complex indexing for corner elements. This is well-documented with inline comments.

#### `src/losses/balancer.py` — Gradient-Norm Balancer

**`GradNormBalancer`:**
- Dynamically balances task loss vs. topographic loss via gradient norms
- Uses EMA smoothing (`beta=0.1`) on the ratio `||∇task|| / ||∇topo||`
- Multiplies topographic loss by `rho * smoothed_ratio` to equalize gradient magnitudes
- Clamps result to `[0, lambda_max]` (default 10000)
- When `rho=0`, returns 0 immediately (no topographic loss)

**`grad_norm(loss, parameters)`** — Computes L2 gradient norm with `retain_graph=True` and `allow_unused=True`.

### 4.4 Training Loop

#### `src/training/train_ce.py`

**`train_one_epoch()`:**
1. Forward pass: `embeddings, logits = model(x)`
2. Task loss: `CrossEntropyLoss(logits, targets)`
3. Topographic loss (if `rho > 0`):
   - `ws`: `Local_WS_Loss` on the model's FC layer weights
   - `global`: `Global_Topographic_Loss` on pre-ReLU embeddings
4. GradNormBalancer produces `lambda_hat` scaling factor
5. Combined loss: `task_loss + lambda_hat * topo_loss`
6. Optional AMP (mixed precision) support via `GradScaler`

**`validate()`:**
- Standard eval loop: loss + accuracy on val/test loader
- Uses `_LogitsOnly` wrapper if model returns (embeddings, logits) tuple

**Helper classes:**
- `AverageMeter` — Running mean/sum/count tracker
- `accuracy()` — Top-k accuracy computation

#### `src/training/checkpoint.py`

Minimal utilities:
- `save_checkpoint(path, state_dict)` — `torch.save` with `os.makedirs`
- `load_checkpoint(path, device)` — `torch.load` with `map_location` and `weights_only=False`

### 4.5 Inference & Caching

#### `src/inference.py`

**`run_combined_model_inference()`:**
- Runs model on loader, collecting: `preds`, `logits`, `probs`, `embeddings`, `labels`
- If model returns only logits (not a tuple), uses logits as embeddings

**`ARTIFACT_KEYS`** = `["logits", "preds", "probs", "embeddings", "labels", "example_ids", "original_indices"]`

**`save_inference_artifacts()`** / **`load_inference_artifacts()`** — Backend-agnostic save/load.

**`artifacts_complete()`** — Checks if minimum required artifacts (`logits`, `preds`, `labels`) exist locally.

**`get_or_run_inference()`** — Main API: load from cache or run inference and save.

### 4.6 Profiling & Metrics

#### `src/profiling/category_similarity.py`

**`compute_similarity_profile(embeddings, anchor_embeddings, metric)`:**
- `cosine`: normalized dot product → `[N, K]` where values ∈ [-1, 1]
- `l2`: negative Euclidean distance → `[N, K]` where values ≤ 0 (higher = more similar)

**`similarity_profile_hash()`** — Deterministic hash from `(run_id, anchor_spec_hash, metric, split)`.

#### `src/profiling/smoothness.py`

**`morans_i(activations, emb_dim)`:**
- Computes Global Moran's I for spatial autocorrelation on a 2D grid
- Averages activations over samples, then measures spatial correlation
- Uses 4-connected (rook) adjacency
- Returns z-score normalized value

**Note:** Uses explicit Python loops for adjacency computation. The code comments acknowledge this is intentional for clarity at current scale (emb_dim ≤ 256).

#### `src/profiling/unit_analysis.py`

- **`weight_norms(fc_layer)`** — L2 norm of each output unit's weight vector
- **`unit_distance_correlation(fc_layer)`** — Returns `[2, N_pairs]` tensor of `(grid_distance, cosine_similarity)` for all unit pairs. Caller computes correlation.

#### `src/profiling/diversity.py`

Comprehensive pairwise diversity metrics for ensembles.

**Architecture:**
```
MetricSpec (name, function, strategy, data_key)
    │
    ├── Strategy.PAIRWISE_COUNTS  → _apply_counts() → _avg_off_diag()
    ├── Strategy.PAIRWISE_LIST    → _apply_list()   → _avg_off_diag()
    └── Strategy.GLOBAL           → direct function call
```

**`AgreementCounts`** — Named tuple with `n11` (both correct), `n00` (both wrong), `n10`, `n01`, plus derived properties (`N`, `accuracy_i`, `accuracy_j`, `p_agree`, etc.).

**`EvalContext`** — Lazy container with `@cached_property` for vectorized agreement count computation (avoids recomputation across metrics).

**Registered metrics:**

| Metric | Strategy | Description |
|--------|----------|-------------|
| `q_statistic` | PAIRWISE_COUNTS | Yule's Q: `(n11·n00 - n10·n01) / (n11·n00 + n10·n01)` |
| `disagreement` | PAIRWISE_COUNTS | `(n10 + n01) / N` |
| `double_fault` | PAIRWISE_COUNTS | `n00 / N` |
| `interrater_agreement` | PAIRWISE_COUNTS | Cohen's κ analog |
| `correlation` | PAIRWISE_COUNTS | `(n11·n00 - n10·n01) / √(prod of marginals)` |
| `iou_top_n` | PAIRWISE_LIST | IoU of top-N confident predictions |

**`compute_metrics(ctx, metric_names, reduce_group)`** — Dispatches to correct strategy, optionally averaging off-diagonal elements.

#### `src/profiling/rdm.py`

- **`pearson_corrcoef(X)`** — Full Pearson correlation matrix for row-vectors
- **`pearson_rdm(X)`** — `1 - pearson_corrcoef(X)` (dissimilarity matrix, diagonal = 0)
- **`upper_triangle_vector(M)`** — Extracts upper triangle as 1D vector (excludes diagonal)
- **`rsa_correlation(rdm_a, rdm_b)`** — RSA: Pearson correlation between upper-triangle vectors of two RDMs

### 4.7 Ensemble Logic

#### `src/ensemble/combine.py`

Four combination methods for logits from M models on N samples:

| Method | Algorithm | Output |
|--------|-----------|--------|
| `soft` | Average softmax probabilities | `[N, C]` probability distribution |
| `hard` | Majority vote on argmax predictions | `[N, C]` one-hot vectors |
| `max_confidence` | Select model with highest max-softmax | `[N, C]` probabilities |
| `conf_weighted` | Weight each model by its confidence | `[N, C]` weighted probabilities |

**Note:** `hard` voting uses a Python loop over N samples. The code comments acknowledge this is intentional for clarity, noting that vectorized alternatives (e.g., `torch.mode`) exist for large N.

#### `src/ensemble/accuracy.py`

- **`ensemble_accuracy(probs, labels)`** — Accuracy of argmax predictions
- **`component_accuracies(logits_list, labels)`** — Per-component and summary (mean, max) accuracies

#### `src/ensemble/selector.py`

**`resolve_components(selector, experiment_name)`:**
- Queries MLflow for FINISHED model runs matching declarative selector predicates
- Supports `eq` (equality), `range` (inclusive range), `in` (membership list)
- MLflow only supports AND-equality filters; `range`/`in` applied as post-filters in Python
- Returns sorted list of run IDs

### 4.8 MLflow Utilities (`src/mlflow_utils.py`)

**Setup:**
- `setup_mlflow(cfg)` — Sets tracking URI, experiment, creates directories, optionally enables system metrics

**Logging:**
- `log_resolved_config(cfg)` — Logs Hydra config YAML as artifact
- `log_git_info()` — Logs `git_commit`, `git_dirty` tags; logs `git_diff.patch` if dirty

**Idempotency:**
- `find_finished_run(experiment, cfg_hash, kind)` — Generic run lookup
- `find_finished_inference_run(experiment, parent_id, split)` — Inference-specific
- `find_finished_behavior_run(experiment, behavior_input_hash, behavior)` — Behavior-specific
- `find_finished_similarity_profile_run(...)` — Profile-specific

**Hashing:**
- `component_set_hash(run_ids)` — SHA-256[:16] of sorted run IDs (JSON)
- `behavior_input_hash(...)` — Composite hash of component set + manifest + split + feature type + anchor spec + meta split + similarity metric + init seed

**Tag builders:**
- `model_tags()` — Standard tags for model training runs
- `behavior_tags()` — Tags for ensemble/adapter behavior runs
- `dataset_manifest_tags()` — Tags for manifest runs
- `profiles_tags()` — Tags for profile runs
- `category_similarity_profile_tags()` — Tags for CSP runs

### 4.9 Config Utilities

See [Section 3](#3-configuration-system) for detailed coverage.

---

## 5. Pipeline Scripts

### `01_train_models.py` — Model Training

**Flow:**
1. Apply schema defaults, resolve seed (`100 + trial`)
2. Compute `cfg_hash` → idempotency check
3. Load CIFAR-10 data (train/val/test) + create dataset manifest
4. Build `LinearResNet18` with `ret_emb=True`
5. Setup losses: `CrossEntropyLoss` + topographic loss (`Local_WS_Loss` or `Global_Topographic_Loss`)
6. Setup `GradNormBalancer` for dynamic loss scaling
7. Training loop: `train_one_epoch()` + `validate()` per epoch
8. Log metrics to MLflow each epoch
9. Periodic + best checkpoint saving
10. Early stopping with configurable patience
11. Final test evaluation on best checkpoint
12. Log best checkpoint as MLflow artifact

**Sweepable via `--multirun`:** `loss.rho`, `loss.topology`, `trial`

### `02_cache_inference.py` — Inference Caching

**Flow:**
1. Find all FINISHED model runs via MLflow query
2. For each run:
   - Check idempotency (MLflow run exists or local cache complete)
   - Download checkpoint, reconstruct model
   - Run inference on eval split
   - Save artifacts locally (logits, preds, probs, embeddings, labels, example_ids, original_indices)
   - Log as MLflow inference run

**Key detail:** `_load_model_from_run()` reconstructs model from MLflow run parameters (`embedding_dim`, `num_classes`, `p_dropout`, `head_bias`).

### `03_compute_profiles.py` — Similarity Profiles

**Flow:**
1. Build `AnchorSpec` from adapter config (not gated on `feature_type`)
2. Get or create anchors from manifest
3. For each model run: compute cosine/L2 similarity profiles against anchor embeddings
4. Save locally + log as MLflow run (`kind=category_similarity_profile`)

**Design decision:** Profiles are always computed for all model runs regardless of `adapter.feature_type`. This is intentional — downstream steps may need profiles even when current config uses `logits` only.

### `03b_compute_diagnostics.py` — Diagnostic Metrics

**Flow:**
1. Check which diagnostics are enabled via `pipeline.diagnostics.*`
2. Per-model, per-metric idempotency: one MLflow run per `(model, metric)`
3. **Moran's I:** Load cached embeddings → compute spatial autocorrelation
4. **Weight norms:** Load checkpoint → L2 norm per output unit
5. **Unit distance correlation:** Load checkpoint → grid distance vs weight cosine

> **Bug fixed:** The script was using an undefined `artifacts_root` variable. See [Section 7](#7-identified-bugs--fixes-applied).

### `04_run_ensemble.py` — Ensemble Voting

**Flow:**
1. Parse ensemble definitions from Hydra config
2. Resolve component run IDs via `selector.resolve_components()`
3. Verify manifest hash compatibility across components
4. Load logits for all components (HARD FAIL on missing)
5. For each vote method: combine logits → compute accuracy → log MLflow run

### `04b_compute_diversity.py` — Diversity Metrics

**Flow:**
1. For each ensemble: resolve components, check idempotency per metric
2. Load predictions + labels for all components
3. Compute metrics via `compute_metrics()` with shared `EvalContext`
4. Log one MLflow run per `(ensemble, metric)`

### `04c_compute_consistency.py` — RSA Consistency

**Flow:**
1. For each ensemble: resolve components, compute per-model RDMs from anchor embeddings
2. Compute pairwise RSA correlation matrix
3. Report mean off-diagonal RSA as consistency measure
4. Save RDM/RSA artifacts + log MLflow run

### `05_train_adapters.py` — Meta-Learner Training

**Flow:**
1. Resolve anchor selection, assemble features based on `feature_type`:
   - `logits`: stacked logits → `[N, M*C]`
   - `embeddings`: stacked embeddings → `[N, M*D]`
   - `embeddings+profiles`: stacked `[embedding | profile]` per model → `[N, M*(D+K)]`
2. Three-way split (60% train, 20% val, 20% holdout) with fixed seed
3. Build adapter (`LinearAdapter` or `TwoLayerMLPAdapter`)
4. Train with Adam, track best validation accuracy
5. Evaluate on holdout set
6. Log adapter weights, component IDs, split membership

**Demand-driven profiles:** If `feature_type=embeddings+profiles`, profiles are computed on-demand if not pre-cached (Step 03 may have been skipped).

---

## 6. Test Suite

**128 tests** across 7 files, all passing. No conftest.py; all setup is local to test files.

### Test file inventory

| File | Tests | Coverage |
|------|-------|----------|
| `test_anchor_determinism.py` | 17 | AnchorSpec freezing, hashing, selection, behavior hash integration |
| `test_cache_alignment.py` | 5 | Manifest save/load, example_id uniqueness, alignment preservation |
| `test_category_similarity.py` | 35 | Profile hashing, cosine/L2 computation, caching, feature dimensions, tags |
| `test_cfg_hash.py` | 14 | Config hash stability, key ordering, exclusion of non-semantic keys |
| `test_ensemble.py` | 12 | combine_logits methods, component_set_hash stability |
| `test_hydra_config.py` | 36 | Hydra composition, group validation, adapter config, EXCLUDED_KEYS |
| `test_profile_gating.py` | 9 | Profile generation not gated on feature_type, skip flag |

### Testing patterns

- **Determinism tests:** Hash stability across calls, order invariance
- **Boundary tests:** Insufficient samples, unknown methods, edge cases
- **Mock-based:** `@patch` and `MagicMock` for MLflow integration tests
- **Fixture-based:** Hydra DictConfig composition for config tests
- **Isolation:** `GlobalHydra.instance().clear()` in test teardown

### Coverage gaps (potential improvements)

- No tests for `src/losses/topographic.py` (topographic loss correctness)
- No tests for `src/losses/balancer.py` (gradient-norm balancer)
- No tests for `src/training/train_ce.py` (training loop)
- No tests for `src/profiling/smoothness.py` (Moran's I)
- No tests for `src/profiling/unit_analysis.py` (weight analysis)
- No tests for `src/profiling/diversity.py` (diversity metrics)
- No tests for `src/profiling/rdm.py` (RDM/RSA)
- No tests for `src/ensemble/selector.py` (component resolution)
- No integration tests for pipeline scripts

---

## 7. Identified Bugs & Fixes Applied

### Bug 1: `NameError` in `03b_compute_diagnostics.py` (CRITICAL)

**File:** `scripts/03b_compute_diagnostics.py`, lines 198, 204  
**Severity:** **Critical** — causes runtime crash (`NameError: name 'artifacts_root' is not defined`)

**Description:** The `main()` function uses `artifacts_root` on lines 198 and 204, but this variable is never defined. Every other script in the pipeline defines `artifacts_root = str(cache_dir)` after obtaining `cache_dir` from `get_cache_dir(cfg)`.

**Root cause:** Likely a copy-paste omission when the script was created.

**Fix applied:**
```python
# Before (broken):
cache_dir = get_cache_dir(cfg)
diag_cfg = cfg.pipeline.diagnostics

# After (fixed):
cache_dir = get_cache_dir(cfg)
artifacts_root = str(cache_dir)
diag_cfg = cfg.pipeline.diagnostics
```

### Bug 2: Misleading class name `ThreeLayerMLPAdapter` (MODERATE)

**File:** `src/networks/heads.py`, line 22  
**Severity:** **Moderate** — misleading name could cause confusion

**Description:** The class `ThreeLayerMLPAdapter` contains only two `nn.Linear` layers:
```python
nn.Linear(in_dim, hidden_dim)  # Layer 1
nn.ReLU(inplace=True)
nn.Dropout(dropout)
nn.Linear(hidden_dim, num_classes, bias=bias)  # Layer 2
```

This is a standard **two-layer** MLP (input → hidden → output). The name "Three-layer" is incorrect — there are only 2 linear transformation layers (activation functions and dropout are not counted as layers in standard MLP terminology).

**Fix applied:** Renamed to `TwoLayerMLPAdapter` across all 4 files:
- `src/networks/heads.py` (definition)
- `src/networks/__init__.py` (re-export)
- `scripts/05_train_adapters.py` (import + usage)

---

## 8. Design Observations & Potential Improvements

### 8.1 Strengths

| Aspect | Observation |
|--------|-------------|
| **Idempotency** | Every pipeline step checks for existing results via MLflow tags/hashes before recomputing. Excellent for long-running experiments. |
| **Determinism** | Content-hashing of examples (manifest), deterministic anchor selection, seeded splits — all ensure reproducibility. |
| **Separation of concerns** | Clean layering: config → data → networks → losses → training → profiling → ensemble. Each module has a single responsibility. |
| **Design patterns** | Registry (models, metrics, storage), Factory (build_model, get_backend), Strategy (diversity metrics), Lazy evaluation (EvalContext). |
| **Config management** | Hydra structured configs provide schema validation at composition time. Excluded keys in cfg_hash prevent spurious recomputation. |
| **MLflow integration** | Comprehensive run tracking with tags, params, metrics, and artifacts. Each pipeline step is a first-class MLflow run. |

### 8.2 Observations and Suggestions

#### 8.2.1 `needs_checkpoint` variable defined but never used

**File:** `scripts/03b_compute_diagnostics.py`, line 158

```python
needs_checkpoint = diag_cfg.weight_norms or diag_cfg.unit_distance_correlation
```

This variable is computed but never referenced. It appears to be an early optimization attempt (to skip checkpoint loading when only Moran's I is needed), but the actual conditional loading is handled per-metric inside the loop. The variable is dead code.

#### 8.2.2 Hard-coded `num_classes=10` in diagnostics

**File:** `scripts/03b_compute_diagnostics.py`, line 81

```python
model = LinearResNet18(emb_dim=emb_dim, num_classes=10, ret_emb=True)
```

The `num_classes` is hard-coded to 10. While this works for CIFAR-10, it doesn't read from the run's params like `02_cache_inference.py` does:
```python
num_classes = int(params.get("num_classes", 10))  # From 02_cache_inference.py
```

**Recommendation:** Use `int(run.data.params.get("num_classes", 10))` for consistency.

#### 8.2.3 `Global_Topographic_Loss.forward` modifies `self.D` in place

**File:** `src/losses/topographic.py`, line 93 (approx.)

```python
self.D = self.D.to(pre_relu.device)
```

This modifies the module's state during forward pass, which could cause issues with multi-GPU setups where the device might change between calls. A safer approach would be:

```python
D = self.D.to(pre_relu.device)
```

However, since the code doesn't use multi-GPU for the loss module itself, this is low-risk.

#### 8.2.4 `inference.py` uses logits as embeddings fallback

**File:** `src/inference.py`, around line 60

When the model returns only logits (not a `(embeddings, logits)` tuple), the code uses logits as embeddings. This could produce misleading results if downstream steps expect true embeddings (e.g., similarity profiles).

**Current behavior:** All models are constructed with `ret_emb=True`, so this fallback should never trigger in normal operation. But it could cause silent errors if a model is configured incorrectly.

#### 8.2.5 Python loop in `combine_logits` hard voting

**File:** `src/ensemble/combine.py`, lines 38–41

```python
for i in range(N):
    votes = per_model_preds[:, i]
    counts = torch.bincount(votes, minlength=C)
    hard_preds[i] = counts.argmax()
```

This Python loop iterates over all N samples. For typical evaluation sets (N ≤ 10k), overhead is negligible. For larger sets, this could be vectorized with `torch.mode()` (which returns the most frequent element).

The code already documents this as an intentional trade-off for clarity.

#### 8.2.6 Ensemble `hard` voting tensor not on correct device

**File:** `src/ensemble/combine.py`, lines 34, 42–43

```python
hard_preds = torch.zeros(N, dtype=torch.long)
# ...
hard_onehot = torch.zeros(N, C)
```

These tensors are created on CPU regardless of where the input `logits_stack` lives. If inputs are on GPU, this could cause a device mismatch. However, ensemble evaluation typically runs on CPU with pre-cached logits, so this is low-risk in practice.

#### 8.2.7 `ZarrBackend` is a stub

**File:** `src/data/cache.py`

The `ZarrBackend` raises `NotImplementedError` for all methods. It's a placeholder for future implementation. No code paths currently use it — the config defaults to `"pt"` backend.

#### 8.2.8 Consistency in `_format_rho` usage

**File:** `src/mlflow_utils.py`, line 136

```python
def _format_rho(rho) -> str:
    return str(float(rho))
```

This function exists to ensure consistent string representation of rho values. However, some scripts manually format rho (e.g., `str(float(cfg.loss.rho))` in `01_train_models.py` line 140) rather than using `_format_rho()`. While the behavior is identical, using the helper everywhere would improve consistency.

### 8.3 Architecture Diagram

```
┌──────────────────────────────────────────────────┐
│                  conf/ (Hydra YAML)              │
│  config.yaml → model + loss + dataset + ...      │
└────────────────────┬─────────────────────────────┘
                     │ compose
                     ▼
┌──────────────────────────────────────────────────┐
│           src/config/structured.py               │
│  Dataclass schemas: ConTopoConfig, ModelConfig,  │
│  LossConfig, DatasetConfig, TrainingConfig, ...  │
└────────────────────┬─────────────────────────────┘
                     │ validate
                     ▼
┌──────────────────────────────────────────────────┐
│             scripts/ (Pipeline Steps)            │
│  01 → 02 → 03/03b → 04/04b/04c → 05            │
└───┬──────┬──────┬──────┬──────┬──────┬───────────┘
    │      │      │      │      │      │
    ▼      ▼      ▼      ▼      ▼      ▼
┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐
│train ││infer ││prof- ││ensem-││diver-││adapt-│
│models││cache ││iles  ││ble   ││sity  ││ers   │
└──┬───┘└──┬───┘└──┬───┘└──┬───┘└──┬───┘└──┬───┘
   │       │       │       │       │       │
   ▼       ▼       ▼       ▼       ▼       ▼
┌──────────────────────────────────────────────────┐
│               MLflow Tracking                    │
│  Runs: model, inference, csp, diagnostics,       │
│        behavior (vote/adapter), diversity,       │
│        consistency                               │
│  Artifacts: checkpoints, logits, profiles, etc.  │
└──────────────────────────────────────────────────┘
```

### 8.4 Data Flow Diagram

```
CIFAR-10 dataset
    │
    ▼
┌─────────────────┐     ┌──────────────────┐
│ DatasetManifest │────▶│  Anchor Selection │
│ (example_ids,   │     │  (per_class=100)  │
│  labels, indices)│    └────────┬──────────┘
└────────┬────────┘              │
         │                       │
         ▼                       ▼
┌────────────────┐    ┌──────────────────────┐
│ Model Training │    │ Similarity Profiles  │
│ (ResNet18 + WS)│    │ [N, K] per model     │
└────────┬───────┘    └──────────┬───────────┘
         │                       │
         ▼                       │
┌────────────────┐               │
│ Cached Inference│              │
│ logits [N,C]   │              │
│ embeddings[N,D]│              │
└────┬───────┬───┘              │
     │       │                   │
     │       ▼                   ▼
     │  ┌────────────────────────────────┐
     │  │     Feature Assembly           │
     │  │ logits:            [N, M*C]    │
     │  │ embeddings:        [N, M*D]    │
     │  │ embeddings+profiles:[N, M*(D+K)]│
     │  └────────────┬───────────────────┘
     │               │
     ▼               ▼
┌──────────┐   ┌────────────┐
│ Ensemble │   │  Adapter   │
│ Voting   │   │  Training  │
│ (soft,   │   │ (Linear/MLP│
│  hard,..)│   │  3-way split)
└──────────┘   └────────────┘
```

---

## 9. Security Considerations

### 9.1 `weights_only=False` in `torch.load`

Multiple files use `torch.load(..., weights_only=False)`:
- `src/data/cache.py:43` (PtBackend.load)
- `src/data/anchors.py:128` (load_anchors)
- `src/data/manifest.py:47` (DatasetManifest.load)
- `src/training/checkpoint.py:18` (load_checkpoint)

This allows arbitrary code execution during deserialization. In a research setting with self-generated artifacts, this is acceptable. However, if artifacts could come from untrusted sources, this would be a vulnerability. Where possible, `weights_only=True` should be used (it's already used in `05_train_adapters.py` for profile loading).

### 9.2 No input validation on MLflow filter strings

`src/ensemble/selector.py` builds MLflow filter strings from selector dict values:
```python
parts.append(f"tags.{key} = '{spec['eq']}'")
```

If selector values contain single quotes or MLflow filter syntax, this could produce malformed queries. In practice, selectors come from Hydra config (controlled input), so injection risk is minimal.

### 9.3 Temporary file handling

Several scripts create temporary files for MLflow artifact logging:
```python
with tempfile.NamedTemporaryFile(..., delete=False) as f:
    # write data
    mlflow.log_artifact(f.name, ...)
    os.unlink(f.name)
```

The `os.unlink()` is inside the `with` block but outside a `try/finally`, meaning if `mlflow.log_artifact()` raises, the temp file won't be cleaned up. This is a minor leak concern for long-running processes.

---

## 10. Dependency Analysis

### Core dependencies (from `pyproject.toml`)

| Package | Purpose |
|---------|---------|
| `torch` / `torchvision` | Neural networks, data loading, transforms |
| `hydra-core` / `omegaconf` | Configuration management |
| `mlflow` | Experiment tracking |
| `numpy` | Numerical computations |
| `scipy` | Scientific computing (used in profiling) |
| `scikit-learn` | Machine learning utilities |
| `matplotlib` | Plotting (not heavily used in core code) |
| `pyyaml` | YAML parsing |
| `gitpython` | Git integration for run tracking |
| `psutil` | System metrics |

### Optional dependencies

| Package | Purpose |
|---------|---------|
| `zarr` | Future storage backend (currently stubbed) |
| `pytest` / `pytest-cov` | Testing (dev group) |

### Python version

Requires Python ≥ 3.10 (uses `match/case` syntax and `X | Y` union types).

---

## 11. Summary

### What ConTopo does well

1. **Reproducibility:** Content-hashed manifests, deterministic anchor selection, seeded splits, and config hashing ensure exact reproducibility.
2. **Idempotency:** Every pipeline step checks for existing results before recomputing, enabling safe re-runs and incremental computation.
3. **Modularity:** Clean separation between data, networks, losses, training, profiling, and ensemble logic.
4. **Extensibility:** Registry and factory patterns make it easy to add new models, loss functions, diversity metrics, and storage backends.
5. **Experiment tracking:** Comprehensive MLflow integration with tags, params, metrics, and artifacts for every pipeline step.

### What was fixed

1. **Critical:** Undefined `artifacts_root` variable in `03b_compute_diagnostics.py` — would crash at runtime.
2. **Naming:** `ThreeLayerMLPAdapter` renamed to `TwoLayerMLPAdapter` to match actual architecture (2 linear layers).

### Areas for potential improvement

1. **Test coverage:** Losses, training loop, profiling metrics, and diversity metrics lack dedicated tests.
2. **Device handling:** Some tensor operations in ensemble code don't respect input device.
3. **Error handling:** Temp file cleanup could use `try/finally` for robustness.
4. **Hard-coded values:** `num_classes=10` in diagnostics script should read from run params.
5. **Dead code:** `needs_checkpoint` variable in diagnostics script is computed but never used.
