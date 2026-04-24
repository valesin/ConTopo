# Datasets

Reference for all datasets supported in ConTopo: what they are, how to set them
up, what training and sweep choices apply to each, and how to add a new one.

---

## 1. Available datasets at a glance

| Dataset | Config key | Classes | Image size | Setup | Backend |
|---|---|---|---|---|---|
| CIFAR-10 | `cifar10` | 10 | 32×32 | auto-download | torch, FFCV |
| Oxford 102 Flowers | `flowers102` | 102 | 224×224 | auto-download | torch |
| ImageNet100 | `imagenet100` | 100 | 224×224 | manual (see §3.3) | torch, FFCV |

---

## 2. Train / val / test split logic

All datasets share the same split mechanism. The factory provides two raw
partitions (`train=True` and `train=False`); a deterministic val subset is then
carved from the training pool at runtime by `_split_train_val_indices` in
`src/data/loaders.py`.

```
train=True pool  ──► first val_per_class samples/class ──► val set
                 └─► remaining samples                  ──► train set
train=False pool ────────────────────────────────────────► test set
```

`val_per_class` is set in each dataset's config under `split.val_per_class`.
The carving is deterministic (based on the dataset's native ordering) so the
same split is produced on every run for the same config.

For datasets with an official val split (e.g. Flowers102), the factory
concatenates the official train and val partitions into the `train=True` pool
so the carving mechanism has enough images per class to work with. See §3.2
for the Flowers102 specifics.

---

## 3. Dataset reference

### 3.1 CIFAR-10

| | |
|---|---|
| **Config** | `conf/dataset/cifar10.yaml` |
| **Classes** | 10 |
| **Image size** | 32×32 |
| **Setup** | Auto-downloads on first run (`download=True`). Data lands under `<data_root>/cifar-10-batches-py/`. |
| **Splits** | Train pool: 50 000 images. Val: 5 000 (500/class carved). Test: 10 000. |
| **Model** | `resnet18` — `LinearResNet18`, 256-dim embedding |
| **Transform** | `cifar10_resizedcrop_v1` — `RandomResizedCrop(32)` + HFlip train; identity eval |
| **Sweeps** | `cifar10_t10`, `cifar10_t20`, `cifar10_torus_t10`, `cifar10_torus_t5`, `cifar10_valloss_t10` |
| **MLflow experiment** | `contopo_cifar10` |

**Quick start:**

```bash
# Smoke-test (downloads dataset on first run):
uv run python scripts/01_train_models.py training.epochs=1

# Full sweep:
python main.py +sweeps=cifar10_t10
```

---

### 3.2 Oxford 102 Flowers (`flowers102`)

| | |
|---|---|
| **Config** | `conf/dataset/flowers102.yaml` |
| **Classes** | 102 |
| **Image size** | 224×224 |
| **Setup** | Auto-downloads on first run (`download=True`). Requires `scipy` for reading `.mat` label files (`pip install scipy` or `uv add scipy`). |
| **Splits (official)** | Train: 1 020 (exactly 10/class). Val: 1 020 (exactly 10/class). Test: 6 149 (~60/class avg, 20/class min). |
| **Splits (pipeline)** | Factory combines official train+val → 2 040 images (20/class pool). Val carving takes 10/class → 1 020 val + 1 020 train. Test is the official test split. |
| **Model** | `resnet34_imagenet100` — `FinetuneResNet34`, pretrained ImageNet1K backbone, 256-dim embedding |
| **Transform** | `flowers102_v1` — `ToTensor()` only; **no augmentation, no normalisation**. Raw pixel values in [0, 1]. |
| **Profiling** | `flowers102` — `per_class=10` (1 020 total anchors), drawn from test split |
| **Sweeps** | `flowers102_t5` |
| **MLflow experiment** | `contopo_flowers102` |
| **Training config** | `epochs=50`, `batch_size=64`, `lr=0.0001` (fine-tuning regime) |

**Quick start:**

```bash
# Smoke-test (downloads dataset on first run — requires scipy):
uv run python scripts/01_train_models.py dataset=flowers102 model=resnet34_imagenet100 profiling=flowers102 training.epochs=1

# Full sweep:
python main.py +sweeps=flowers102_t5
```

**Note on the split design:** Flowers102 has only 10 images/class in each
official split, which is too small to carve a val subset from the training
split alone. The factory therefore concatenates the official train and val
splits (20/class) and lets the pipeline carve 10/class as validation. This
leaves 10/class for training — a deliberately small training set, consistent
with the dataset's intended use as a fine-tuning benchmark.

---

### 3.3 ImageNet100 (`imagenet100`)

| | |
|---|---|
| **Config** | `conf/dataset/imagenet100.yaml` |
| **Classes** | 100 (subset of ImageNet) |
| **Image size** | 224×224 |
| **Setup** | **Manual** — data must be on disk before running (see layout below). |
| **Splits** | Train pool: 50 000 (500/class). Val: 5 000 (50/class carved). Test: 5 000 (official `val/` dir). |
| **Models** | `resnet34_imagenet100` — `FinetuneResNet34`, pretrained backbone *(default)*<br>`resnet34_scratch` — `ScratchResNet34`, random init |
| **Transform** | `imagenet_v1` — `RandomResizedCrop(224)` + HFlip train; `Resize(256)` + `CenterCrop(224)` eval. ImageNet1K V1 normalisation. |
| **Profiling** | `imagenet100` — `per_class=10` (1 000 total anchors), drawn from test split |
| **Sweeps** | `imagenet100_t10`, `imagenet100_t5`, `imagenet100_ffcv_t10`, `imagenet100_ffcv_t5` |
| **MLflow experiment** | `contopo_imagenet100` (torch) / `contopo_imagenet100_ffcv` (FFCV) |
| **Training config (torch)** | `epochs=30`, `batch_size=64`, `lr=0.0001` (fine-tuning) |
| **Training config (FFCV)** | `epochs=16`, `batch_size=1024`, `lr=0.5`, SGD + OneCycleLR, label smoothing, blurpool, TTA, progressive resolution 160→192px |

**Data layout:**

```
<runtime.data_root>/
└── imagenet100/
    ├── train/
    │   ├── n01440764/   ← WordNet synset folder (one per class)
    │   └── ...          ← 100 class folders × 500 images = 50 000 images
    └── val/
        ├── n01440764/
        └── ...          ← 100 class folders × 50 images = 5 000 images
```

`val/` is the **held-out test split** — never touched during training or val
carving. A symlink from a canonical data location is fine:

```bash
ln -s /path/to/canonical/imagenet100 <data_root>/imagenet100
```

**Quick start:**

```bash
# Smoke-test:
uv run python scripts/01_train_models.py \
    dataset=imagenet100 model=resnet34_imagenet100 profiling=imagenet100 \
    mlflow.experiment_name=contopo_imagenet100 training.epochs=1

# Full torch sweep:
python main.py +sweeps=imagenet100_t10

# Full FFCV sweep (requires ffcv install — see docs/ffcv_param_assumptions.md):
python main.py +sweeps=imagenet100_ffcv_t10
```

**Model choice:** use `resnet34_imagenet100` (fine-tuning) for the standard
experiment. Use `resnet34_scratch` to study the effect of random initialisation;
note that training from scratch on 50 000 images typically requires more epochs
and a higher learning rate than the fine-tuning defaults.

---

## 4. Adding a new dataset: torch pipeline

### Step 1 — Write the factory

Every dataset is registered via a factory with this signature:

```python
def _<name>_factory(
    root: str,
    train: bool,
    transform: Callable | None,
    download: bool = ...,
) -> Dataset:   # must expose .targets: list[int]
```

| Argument | Meaning |
|---|---|
| `root` | `cfg.runtime.data_root` — the base data directory |
| `train=True` | Training-pool partition (val is carved from this) |
| `train=False` | Held-out test partition |
| `transform` | Callers pass `None` when building index structures |

The returned object **must** expose `.targets` — a flat Python list of integer
class labels, one per sample.

**Standard torchvision API** (`train: bool`) — thin wrapper:

```python
def _mydata_factory(root, train, transform, download=False):
    return datasets.MyData(root=root, train=train, transform=transform, download=download)
```

**Split-string API** (`split='train'|'val'|'test'`) — adapt to the bool
contract and attach `.targets` explicitly (Flowers102 is the worked example;
see `src/data/loaders.py:_flowers102_factory`).

**ImageFolder-based datasets** — map `train` to the appropriate subdirectory:

```python
def _mydata_factory(root, train, transform, download=False):
    subset = "train" if train else "val"
    return datasets.ImageFolder(root=os.path.join(root, "mydata", subset), transform=transform)
```

`ImageFolder` exposes `.targets` natively.

### Step 2 — Register

In `src/data/loaders.py`, add to both dicts:

```python
_DATASET_FACTORIES = { ..., "mydata": _mydata_factory }
DATASET_NUM_CLASSES = { ..., "mydata": <N> }
```

### Step 3 — Add a transform preset

Named presets live in `src/data/transforms.py`. Names are **versioned** —
never mutate an existing preset, create a new version instead. The preset name
feeds `cfg_hash` via `dataset.transforms.preset`.

```python
def _mydata_v1():
    train = transforms.Compose([...])
    eval_ = transforms.Compose([...])
    return train, eval_

_PRESETS = { ..., "mydata_v1": _mydata_v1 }
```

Be explicit in the docstring about what augmentation and normalisation is applied.
If the preset is intentionally minimal (e.g. `ToTensor()` only), say so.

### Step 4 — Create the Hydra config

`conf/dataset/mydata.yaml`:

```yaml
name: mydata
image_size: <H>
in_channels: 3
mean: [R, G, B]   # set to [0,0,0] if the preset does not normalise
std:  [R, G, B]   # set to [1,1,1] if the preset does not normalise
split:
  strategy: first_n_per_class
  val_per_class: <N>   # ≤ images/class in the train=True pool
transforms:
  preset: mydata_v1
```

`mean` and `std` are required by the schema even when the torch preset does
not use them: they feed the FFCV `NormalizeImage` kernel if that backend is
ever activated.

### Step 5 — Create a profiling config

`conf/profiling/mydata.yaml` — set `per_class` to a value that is safely below
the minimum test-split count per class:

```yaml
anchors:
  per_class: <N>
  strategy: per_class_first_n
  order_by: example_id
  source_split: test

profiles:
  skip: false
  metrics: [cosine]

diagnostics:
  morans_i: true
  weight_norms: true
  unit_distance_correlation: true
```

### Step 6 — Add a sweep

`conf/sweeps/mydata_t<N>.yaml` — follow the pattern of any existing production
sweep. Minimum required fields:

```yaml
# @package _global_
defaults:
  - override /dataset: mydata
  - override /model: <model>
  - override /profiling: mydata

mlflow:
  experiment_name: contopo_mydata

hydra:
  mode: MULTIRUN
  sweeper:
    params:
      trial: range(0, <N>)
      loss.rho: 0.0, 0.008, 0.04, 0.2, 1.0, 5.0
      loss.topology: torus, grid
```

### Step 7 — Verify

```bash
uv run python scripts/01_train_models.py \
    dataset=mydata model=<model> profiling=mydata \
    mlflow.experiment_name=contopo_mydata \
    training.epochs=1
```

---

## 5. Adding a new dataset: FFCV pipeline

No additional code is needed beyond a correct torch factory (§4 steps 1–2),
provided the factory's `__getitem__` returns `(PIL Image, int label)` tuples —
which is the default for all standard torchvision datasets.

`beton_writer.get_or_write_beton` calls the factory with `transform=None` and
writes a `.beton` file once; the same file is reused across all runs with the
same (dataset, split, beton config) combination.

**FFCV-specific config fields** (validated at startup, fail if absent when
`loading_backend=ffcv`):

| Field | Example value | Notes |
|---|---|---|
| `training.loading_backend` | `"ffcv"` | Triggers FFCV path |
| `training.beton.max_resolution` | `500` | Max stored pixel size |
| `training.beton.jpeg_quality` | `90` | JPEG compression quality |
| `training.beton.compress_probability` | `0.50` | Fraction of images compressed |

`ConcatDataset`-backed factories (e.g. Flowers102) work transparently: the
beton writer wraps the result in `torch.utils.data.Subset`, which delegates
`__getitem__` and `__len__` through to the underlying dataset.

See `docs/ffcv_param_assumptions.md` for FFCV dependency install, pipeline
implementation notes, and migration history.

---

## Reference

| File | Role |
|---|---|
| `src/data/loaders.py` | Factory registry, `_split_train_val_indices`, loader dispatch |
| `src/data/transforms.py` | Named transform presets |
| `src/data/beton_writer.py` | On-demand `.beton` file generation |
| `src/data/ffcv_pipelines.py` | FFCV augmentation pipeline builders |
| `conf/dataset/` | Per-dataset Hydra configs |
| `conf/profiling/` | Per-dataset profiling configs |
| `conf/sweeps/` | Sweep definitions |
| `docs/ffcv_param_assumptions.md` | FFCV config fields, dependencies, migration history |
