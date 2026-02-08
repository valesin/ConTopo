# ConTopo: Contrastive Learning with Topographic Regularization

> **Paper**: *Similar Accuracy but Different [Structure]: How Training Objectives Shape Topographic Representations*

A PyTorch framework for studying how **topographic regularization** interacts with different training objectives (cross-entropy, SupCon, SimCLR, margin-based contrastive) in neural networks. The topographic constraint encourages spatially adjacent units to have similar response profiles, mimicking cortical spatial organization.

## Research Overview

This codebase replicates the study examining:
1. **Task performance** under varying topographic regularization strengths (ρ)
2. **Topographic smoothness** of activations (Moran's I)
3. **Functional co-localization** of correlated units
4. **Representational geometry** and cross-seed consistency (RDM/RSA)

### Key Findings
- Supervised objectives (CE, margin, SupCon) maintain ~91% accuracy across all ρ values
- Topographic constraints reduce cross-seed representational consistency
- Different objectives produce distinct spatial kernel properties
- Dropout (p=0.5) is critical for achieving cortex-like smooth representations

---

## Replicating the Paper Results

### Step 1: Environment Setup

```bash
pip install torch torchvision tensorboard scikit-learn matplotlib pyyaml numpy
```

### Step 2: Full Training Grid

The paper evaluates **24 model types**: 4 objectives × 6 ρ values, each with 5 random seeds.

**Run all experiments automatically:**
```bash
python run_all.py
```

This executes all grids defined in `configs/experiments.json`:
- 4 objectives × 6 ρ values × 5 trials = 120 runs (with dropout)
- 4 objectives × 6 ρ values × 1 trial = 24 runs (no-dropout ablation)

Expected runtime: several days on a single GPU.

**Or run individual conditions:**

```bash
# Cross-Entropy (CE)
python main_ce.py ws resnet18 --epochs 200 --topographic_loss_rho 0.2 --trial 0

# SupCon (Supervised Contrastive)
python main_supcon.py ws resnet18 --epochs 500 --topographic_loss_rho 0.2 --trial 0

# SimCLR (Self-Supervised Contrastive)
python main_supcon.py ws resnet18 --epochs 800 --task_method simclr --topographic_loss_rho 0.2 --trial 0

# Margin-based Contrastive (with animacy hierarchy)
python main_coscontr.py ws resnet18 --epochs 500 --topographic_loss_rho 0.2 --trial 0
```

### Step 3: The ρ (Rho) Grid

The paper uses a **gradient-matched** weighting scheme where ρ controls the relative strength of topographic vs. task gradients:

| ρ Value | Interpretation |
|---------|----------------|
| 0 | No topographic constraint (baseline) |
| 0.008 | Very weak topography |
| 0.04 | Weak topography |
| **0.2** | Moderate topography (main comparison) |
| 1.0 | Equal task/topo gradient magnitude |
| 5.0 | Strong topographic dominance |

### Step 4: Run Analysis Experiments

After training, run the analysis scripts on the saved models:

```bash
# Smoothness metric (Moran's I)
python exp_smoothness.py ./save/ResNet18/models/<model_folder>/

# Unit distance analysis by correlation threshold
python exp_unitdist.py ./save/ResNet18/models/<model_folder>/

# Generate RDMs (Representational Dissimilarity Matrices)
python exp_generateRDM.py ./save/ResNet18/models/<model_folder>/

# RSA (Representational Similarity Analysis) across models
python exp_RSA.py ./save/ResNet18/models/

# Error correlation and ensemble analysis (with noise robustness)
python exp_errorcorr.py ./save/ResNet18/models/<model_folder>/

# t-SNE visualization
python exp_tsne.py ./save/ResNet18/models/<model_folder>/

# Activation maps per class
python exp_actmaps.py ./save/ResNet18/models/<model_folder>/

# Weight norm statistics
python exp_weightnorms.py ./save/ResNet18/models/<model_folder>/
```

**Run analysis across all models in a directory:**
```bash
python run_all_experiments.py exp_generateRDM.py ./save/ResNet18/models/ --log-dir ./logs
```

---

## Key Training Parameters (Paper Defaults)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Architecture | ResNet18 (modified) | 3×3 initial conv, stride 1, no max-pool |
| Embedding dim | 256 | Arranged on 16×16 grid |
| Projection dim | 128 | For contrastive objectives |
| Batch size | 512 | Required for contrastive learning |
| Optimizer | Adam | lr=0.002 |
| Dropout | p=0.5 | **Critical for smoothness** |
| Early stopping | patience=25 | Based on validation loss/accuracy |
| Validation split | 5000 | 500 per class from training set |

### Contrastive-Specific Settings

| Parameter | Value |
|-----------|-------|
| Readout epochs | 200 |
| Readout optimizer | AdamW |
| Readout lr | 0.003 |
| Readout weight decay | 0.01 |
| Augmentations | RandomResizedCrop, HFlip, ColorJitter, RandomGrayscale |

### Margin Loss Hyperparameters

| Parameter | CLI | Default | Description |
|-----------|-----|---------|-------------|
| `margin_same` (mñ) | `--margin_same` | 0.3 | Negative pairs, same animacy |
| `margin_diff` (mn) | `--margin_diff` | 0.5 | Negative pairs, different animacy |
| `posdist_margin` (mp) | *hardcoded* | 0.05 | Positive pairs (in `CosineContrastiveLoss`)|

---

## Directory Structure

```
ConTopo/
├── main_ce.py            # Cross-entropy training
├── main_supcon.py        # SupCon / SimCLR training
├── main_coscontr.py      # Margin-based contrastive training
├── run_all.py            # Batch runner (reads configs/experiments.json)
├── run_all_experiments.py# Run exp_* scripts across model folders
│
├── exp_*.py              # Analysis scripts
│   ├── exp_generateRDM.py   # RDM computation
│   ├── exp_RSA.py           # Cross-model RSA
│   ├── exp_errorcorr.py     # Error correlation + noise robustness
│   ├── exp_tsne.py          # t-SNE visualization
│   ├── exp_actmaps.py       # Activation map visualization
│   ├── exp_smoothness.py    # Moran's I (spatial autocorrelation)
│   ├── exp_unitdist.py      # Unit distance by correlation
│   └── exp_weightnorms.py   # FC weight norm statistics
│
├── losses/
│   ├── topographic.py       # Global_Topographic_Loss, Local_WS_Loss
│   ├── supcon.py            # SupConLoss
│   └── cosine_contrastive.py# CosineContrastiveLoss (margin-based)
│
├── networks/
│   ├── modified_ResNet18.py # ResNet18, ProjectionResNet18, LinearResNet18
│   └── shallowCNN.py        # ShallowCNN variants (alternative backbone)
│
├── utils/
│   ├── load.py              # Unified checkpoint loading
│   ├── train.py             # Training utilities, TwoCropTransform
│   └── experiments.py       # CIFAR-10 loader, figure path resolution
│
└── configs/
    ├── cifar10.yaml         # Class names + animacy mappings
    └── experiments.json     # Full experiment grid configuration
```

---

## Topographic Losses

Two topographic loss types are available (`topography_type` positional argument):

### `ws` - Local Weight Smoothness (Paper Default)

The paper uses the **local weight-similarity constraint** (Truong & Hasson 2025):

```
Ltopo = Σ_p Σ_{q ∈ N(p)} ||w_p - w_q||₂
```

- Units are arranged on a 16×16 grid (256 embedding dimensions)
- N(p) = 4-connected neighbors plus diagonals (horizontal, vertical, diagonal down-right, diagonal down-left)
- Penalizes differences between incoming weight vectors of adjacent units

### `global` - Global Topographic Loss

Encourages cosine similarity between unit activations to decay with spatial distance:

```
target_ij ≈ 1 / (d_ij + 1)
Ltopo = Σ_{i<j} (cos_sim(unit_i, unit_j) - target_ij)²
```

### Gradient-Matched Weighting

Both losses use a **gradient-matched** λ that normalizes loss scales:

```
λ* = ρ × ||∇θ Ltask|| / (||∇θ Ltopo|| + ε)
λ̂ = EMA(λ*, β=0.1)
L = Ltask + λ̂ × Ltopo
```

This ensures ρ controls the **relative gradient magnitude** rather than raw loss scale.

---

## Output Structure

Training produces:
```
save/<Arch>/models/<model_name>/trial_XX/
├── e2e_best.pth / e2e_last.pth          # CE checkpoints
├── contrastive_best.pth / contrastive_last.pth  # Encoder (contrastive)
└── readout_best.pth / readout_last.pth  # Linear classifier (contrastive)
```

TensorBoard logs:
```
save/<Arch>/tensorboard/<method>/<model_name>/trial_XX/
```

Figures:
```
save/<Arch>/figures/<experiment>/<model_name>/trial_XX/
```

---

## Loading Trained Models

```python
from utils.load import load_model_bundles

# Load all trials from a model folder
bundles = load_model_bundles("./save/ResNet18/models/supcon_wstopo_.../")

for b in bundles:
    encoder = b.encoder       # Backbone
    classifier = b.classifier # Linear head
    meta = b.meta             # epoch, stage, args, metrics
```

---

## Expected Results (Table 1 from Paper)

| ρ | CE | Margin | SimCLR | SupCon |
|---|---|---|---|---|
| 0 | 0.91 ± 0.00 | 0.90 ± 0.00 | 0.77 ± 0.02 | 0.91 ± 0.00 |
| 0.2 | 0.91 ± 0.01 | 0.91 ± 0.00 | 0.64 ± 0.01 | 0.91 ± 0.01 |
| 5 | 0.91 ± 0.00 | 0.88 ± 0.01 | 0.56 ± 0.01 | 0.91 ± 0.00 |

*Supervised objectives maintain accuracy; SimCLR degrades with increasing ρ.*

---

## CLI Reference

### Training Scripts

**Positional arguments (required):**
| Argument | Choices | Description |
|----------|---------|-------------|
| `topography_type` | `global`, `ws` | Topographic loss type (paper uses `ws`) |
| `model_type` | `shallowcnn`, `resnet18` | Network architecture |

**Common options (all training scripts):**
| Option | Default | Description |
|--------|---------|-------------|
| `--topographic_loss_rho` | 0.05 | Topographic constraint strength ρ |
| `--epochs` | 125 (CE) / 250 (SupCon/Margin) | Training epochs |
| `--batch_size` | 512 | Batch size |
| `--learning_rate` | 0.002 | Learning rate for Adam optimizer |
| `--embedding_dim` | 256 | Embedding dimension (16×16 grid) |
| `--trial` | 0 | Trial number (0-4 for 5 seeds) |
| `--num_workers` | 2 | DataLoader workers |
| `--use_dropout` / `--no-use_dropout` | True | Enable dropout |
| `--p_dropout` | 0.5 | Dropout probability |
| `--print_freq` | 10 | Print frequency during training |

**SupCon/SimCLR specific (`main_supcon.py`):**
| Option | Default | Description |
|--------|---------|-------------|
| `--task_method` | supcon | `supcon` or `simclr` |
| `--projection_dim` | 128 | Projection head output |
| `--readout_epochs` | 200 | Linear readout training epochs |
| `--readout_batch_size` | 2048 | Batch size for readout |
| `--readout_lr` | 0.003 | Learning rate for readout (AdamW) |
| `--readout_weight_decay` | 0.01 | Weight decay for readout |
| `--readout_warmup_epochs` | 3 | Warmup epochs for readout scheduler |
| `--readout_min_lr` | 1e-5 | Final LR after cosine decay |

**Margin loss specific (`main_coscontr.py`):**
| Option | Default | Description |
|--------|---------|-------------|
| `--margin_same` | 0.3 | Margin for same-animacy negative pairs |
| `--margin_diff` | 0.5 | Margin for different-animacy negative pairs |
| `--projection_dim` | 128 | Projection head output |
| `--readout_epochs` | 200 | Linear readout training epochs |
| `--readout_batch_size` | 2048 | Batch size for readout |
| `--readout_lr` | 0.003 | Learning rate for readout (AdamW) |
| `--readout_weight_decay` | 0.01 | Weight decay for readout |
| `--readout_warmup_epochs` | 3 | Warmup epochs for readout scheduler |
| `--readout_min_lr` | 1e-5 | Final LR after cosine decay |

### Analysis Scripts

All `exp_*.py` scripts use a shared argument parser from `utils/load.py`:

```bash
python exp_<name>.py <path> [--prefer best|last] [--device cuda|cpu] [--dp] [--batch-size 256] [--num-workers 8] [--dataset-root ./dataset]
```

| Option | Default | Description |
|--------|---------|-------------|
| `path` | (required) | Path to checkpoint file or run/model folder |
| `--prefer` | best | Choose `best` or `last` checkpoint |
| `--device` | cuda/cpu | Device to load model on |
| `--dp` | False | Wrap encoder in DataParallel if multiple GPUs |
| `--batch-size` | 256 | Batch size for evaluation |
| `--num-workers` | 8 | DataLoader workers |
| `--dataset-root` | ./dataset | Path to CIFAR-10 dataset |

---

## Network Architectures

### ResNet18 (Modified for CIFAR-10)

Located in `networks/modified_ResNet18.py`:

- **ResNet18**: Base encoder with 3×3 initial conv (stride 1, no max-pool), outputs `emb_dim` features
- **LinearResNet18**: Encoder + dropout + linear classifier for cross-entropy training
- **ProjectionResNet18**: Encoder + BatchNorm + ReLU + Dropout + projection head for contrastive training

### ShallowCNN (Alternative Backbone)

Located in `networks/shallowCNN.py`:

- **ShallowCNN**: 4-layer CNN backbone with BatchNorm and max pooling
- **LinearShallowCNN**: ShallowCNN + dropout + linear classifier
- **ProjectionShallowCNN**: ShallowCNN + projection head for contrastive learning
- **LinearClassifier**: Standalone linear classifier for readout

---

## Loss Functions

### SupConLoss (`losses/supcon.py`)

Supervised Contrastive Learning loss with:
- Temperature: 0.07 (default)
- Base temperature: 0.07
- Contrast mode: 'all' (uses all views as anchors)
- Supports both supervised (with labels) and unsupervised (SimCLR) modes

### CosineContrastiveLoss (`losses/cosine_contrastive.py`)

Margin-based contrastive loss with animacy-aware margins:
- Positive pairs: penalize cosine distance > 0.05 (`posdist_margin`)
- Negative pairs (same animacy): enforce margin of 0.3 (`margin_same`)
- Negative pairs (different animacy): enforce margin of 0.5 (`margin_diff`)

### Topographic Losses (`losses/topographic.py`)

- **Global_Topographic_Loss**: Activation-based, penalizes deviation from distance-decayed similarity
- **Local_WS_Loss**: Weight-based, penalizes L2 differences between neighboring unit weight vectors

---

## Utility Modules

### `utils/load.py`

Unified model loading utilities:
- `load_model_bundles()`: Load all trials from a model folder
- `load_encoder_from_ckpt()`: Load encoder from checkpoint file
- `load_encoder_from_run_folder()`: Load encoder from run folder
- `LoadedModelBundle`: Dataclass with `encoder`, `classifier`, and `meta` fields
- `parse_model_load_args()`: Shared argument parser for analysis scripts

### `utils/train.py`

Training utilities:
- `TwoCropTransform`: Apply same transform to two crop views
- `AverageMeter`: Track running averages
- `accuracy()`: Compute top-k accuracy
- `grad_norm()`: Compute gradient L2 norm for gradient-matched weighting
- `split_cifar10_train_val_indices()`: Deterministic 45k/5k train/val split
- `load_cifar10_metadata()`: Load class names and animacy mappings from YAML

### `utils/experiments.py`

Experiment utilities:
- `get_cifar10_eval_loader()`: Build evaluation DataLoader
- `resolve_figure_path()`: Resolve output path for figures based on checkpoint path

---

## Configuration Files

### `configs/cifar10.yaml`

CIFAR-10 class metadata:
```yaml
CIFAR10_CLASSES:
  - airplane
  - automobile
  - bird
  - cat
  - deer
  - dog
  - frog
  - horse
  - ship
  - truck

ANIMACY:
  airplane: inanimate
  automobile: inanimate
  ship: inanimate
  truck: inanimate
  bird: animate
  cat: animate
  deer: animate
  dog: animate
  frog: animate
  horse: animate

ANIMACY_MAPPING:
  animate: 0
  inanimate: 1
```

### `configs/experiments.json`

Full experiment grid configuration with:
- 4 main objectives: cross-entropy, cosine-contrastive, supcon, simclr
- 4 no-dropout ablations
- Each with 6 ρ values: 0.0, 0.008, 0.04, 0.2, 1.0, 5.0
- 5 trials for main experiments, 1 trial for ablations

---

## Analysis Scripts Reference

| Script | Purpose | Key Output |
|--------|---------|------------|
| `exp_smoothness.py` | Compute Moran's I spatial autocorrelation | Mean ± SEM across trials |
| `exp_unitdist.py` | Mean grid distance for correlated unit pairs | Distance per threshold (0.1–0.8) |
| `exp_generateRDM.py` | Generate Representational Dissimilarity Matrices | RDM plots + per-trial tensors |
| `exp_RSA.py` | Cross-model RSA comparison | Correlation heatmaps + CSV |
| `exp_errorcorr.py` | Pairwise error correlations + noise robustness | Correlation matrix + ensemble accuracy |
| `exp_tsne.py` | t-SNE visualization of embeddings | 2D scatter plots |
| `exp_actmaps.py` | Per-class activation map visualization | Heatmaps of FC activations |
| `exp_weightnorms.py` | FC layer weight norm statistics | Mean ± std across trials |

---

## Citation

If you use this codebase, please cite:

```bibtex
@article{similar_accuracy_different,
  title={Similar Accuracy but Different [Structure]: How Training Objectives Shape Topographic Representations},
  year={2025}
}

@article{khosla2020supervised,
  title={Supervised Contrastive Learning},
  author={Khosla, Prannay and others},
  journal={NeurIPS},
  year={2020}
}
```
