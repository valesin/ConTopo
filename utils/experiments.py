import yaml
import os
from typing import Any, Dict
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pathlib import Path
import re

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to the YAML file.
        
    Returns:
        Dictionary containing the configuration.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Validation / defaults could go here
    return config

def get_cifar10_eval_loader(
    root: str = "./dataset",
    batch_size: int = 256,
    num_workers: int = 2,
    pin_memory: bool | None = None,
    drop_last: bool = False,
    subset: int | None = None,
):
    """
    Build a DataLoader for CIFAR-10 *test* split (eval only).

    - Normalizes with CIFAR-10 mean/std.
    - `pin_memory` defaults to CUDA availability.
    - `subset` lets you evaluate on the first N examples (quick checks).
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
             # default or inferred from absence? usually if wstopo is present but

    ds = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    # Optional quick subset for faster eval/plots
    if subset is not None and subset < len(ds):
        ds = Subset(ds, range(subset))

    # Deterministic order for evaluation
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return loader

def resolve_figure_path(src_path: str, experiment: str | None = None) -> str:
    """
    Resolve an output path for a figure based on a checkpoint or model/run directory.

    Layout:
      <Arch>/figures/<experiment?>/<model_name>/<run_name>/<ckpt_basename>.png

    Where:
      - <Arch> is the parent directory of 'models' (e.g., 'ShallowCNN' or 'ResNet18').
      - <model_name> is the first folder under 'models' (your hyperparam folder).
      - <run_name> is the next folder under 'models' (e.g., 'trial_00').
      - <ckpt_basename> is the checkpoint filename without extension (e.g., 'e2e_epoch0200').

    Fallback when no 'models' ancestor exists:
      <run_root>/figures/<experiment?>/<model_dir>/<run_dir>/<ckpt_basename>.png

    Returns the absolute path as string, creating parent directories if needed.
    """
    p = Path(src_path).resolve()

    # If a file is given (ckpt), use its parent as the run directory and its stem as the figure name.
    is_file = bool(p.suffix)
    run_path = p.parent if is_file else p
    ckpt_stem = p.stem if is_file else "model"

    # Find nearest ancestor named 'models'
    models_dir = None
    for parent in [run_path] + list(run_path.parents):
        if parent.name == "models":
            models_dir = parent
            break

    if models_dir is not None:
        # figures root sits next to 'models', i.e. under the architecture folder
        arch_dir = models_dir.parent  # e.g., .../ShallowCNN or .../ResNet18
        figures_root = arch_dir / "figures"
        try:
            # Expect run_path like: <...>/models/<model_name>/<run_name>
            rel = run_path.relative_to(models_dir)
            parts = rel.parts
            model_name = parts[0] if len(parts) >= 1 else run_path.name
            run_name   = parts[1] if len(parts) >= 2 else run_path.name
        except ValueError:
            # If relative_to fails, fall back to directory names
            model_name = run_path.parent.name
            run_name   = run_path.name
    else:
        # No 'models' ancestor → keep figures next to the provided path
        figures_root = run_path / "figures"
        model_name = run_path.parent.name  # hyperparam folder
        run_name   = run_path.name         # run folder

    # Optional experiment subfolder (sanitize to safe filename)
    if experiment and experiment.strip():
        safe_exp = re.sub(r"[^\w.\-]+", "_", experiment.strip())
        out_dir = figures_root / safe_exp / model_name / run_name
    else:
        out_dir = figures_root / model_name / run_name

    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{ckpt_stem}.png")


def parse_model_name(path: str) -> dict:
    """
    Extract hyperparameters from a model folder path.

    Handles standard naming patterns like:
      - crossentropy_wstopo_256embdims_0.05rho_125epochs_512bsz_2nwork_0.002lr_0.5dropout
      - simclr_wstopo_grid_256embdims_128projdims_0.05rho_250epochs_512bsz_nwork8_readep200_lr0.002_0.5dropout

    Returns a dict with extracted values (e.g., {'loss': 'crossentropy', 'rho': 0.05, ...}).
    """
    import os
    import re
    
    name = os.path.basename(os.path.normpath(path))
    info = {'model_name': name}

    # Extract Loss Type (first part)
    if name.startswith('crossentropy_'):
        info['loss'] = 'crossentropy'
    elif name.startswith('supcon_'):
        info['loss'] = 'supcon'
    elif name.startswith('simclr_'):
        info['loss'] = 'simclr'
    else:
        info['loss'] = 'unknown'

    # Extract Topology Type
    if 'wstopo' in name:
        info['topo_type'] = 'ws'
        if '_grid_' in name:
            info['topology'] = 'grid'
        elif '_torus_' in name:
            info['topology'] = 'torus'
        else:
             # default or inferred from absence? usually if wstopo is present but no grid/torus, checks args
             # But let's look for specific patterns
             pass  
    elif 'globaltopo' in name:
        info['topo_type'] = 'global'
    else:
        info['topo_type'] = 'none'

    # Regex patterns for common params
    patterns = {
        'rho': r'([\d\.]+)rho',
        'epochs': r'(\d+)epochs',
        'batch_size': r'(\d+)bsz',
        'lr': r'([\d\.]+)lr',
        'dropout': r'([\d\.]+)dropout',
        'emb_dim': r'(\d+)embdims',
        'proj_dim': r'(\d+)projdims',
        'readout_epochs': r'readep(\d+)',
    }

    for key, pat in patterns.items():
        match = re.search(pat, name)
        if match:
             try:
                # Try converting to int first, then float
                val_str = match.group(1)
                if '.' in val_str:
                    info[key] = float(val_str)
                else:
                    info[key] = int(val_str)
             except ValueError:
                 info[key] = match.group(1)
    
    return info

def select_deterministic_cifar10_subset(val_loader, per_class: int = 100):
    """
    Deterministically collect exactly `per_class` samples for each of the 10 CIFAR-10 classes
    from the evaluation loader (which is ordered and not shuffled).

    Returns images stacked in CLASS-GROUPED order ([1000, C, H, W]) and labels where the
    first 100 belong to class 0, next 100 to class 1, ..., last 100 to class 9.
    """
    imgs_by_class = {i: [] for i in range(10)}

    with torch.no_grad():
        for imgs, labs in val_loader:
            for img, lab in zip(imgs, labs):
                c = int(lab)
                lst = imgs_by_class[c]
                if len(lst) < per_class:
                    lst.append(img)
            # Early exit if all classes are filled
            if all(len(lst) >= per_class for lst in imgs_by_class.values()):
                break

    # Verify and stack in class order 0..9
    for c in range(10):
        if len(imgs_by_class[c]) < per_class:
            raise RuntimeError(f"Could not collect required samples for class {c}: "
                               f"got {len(imgs_by_class[c])}, need {per_class}")

    ordered_imgs = []
    ordered_labels = []
    for c in range(10):
        ordered_imgs.extend(imgs_by_class[c][:per_class])
        ordered_labels.extend([c] * per_class)

    stacked = torch.stack(ordered_imgs, dim=0)
    return stacked, ordered_labels

def compute_embeddings(encoder: torch.nn.Module, images: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    """
    Run images through encoder in batches and return a [N, D] tensor of embeddings (CPU float32).
    """
    encoder.eval()
    feats = []
    N = images.size(0)
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = images[i:i+batch_size].to(device, non_blocking=True)
            out = encoder(batch)
            if out.ndim > 2:
                out = out.flatten(1)
            feats.append(out.detach().cpu().to(dtype=torch.float32))
    return torch.cat(feats, dim=0)


def pearson_rdm(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute 1 - Pearson correlation matrix for row-vectors in X (shape [N, D]).
    Returns a [N, N] tensor on CPU.
    """
    X = X.to(dtype=torch.float32, device="cpu")
    Xc = X - X.mean(dim=1, keepdim=True)
    norms = Xc.norm(dim=1, keepdim=True).clamp_min(eps)
    Y = Xc / norms
    corr = Y @ Y.t()
    rdm = 1.0 - corr
    # Ensure perfect self-similarity maps to 0 exactly
    rdm.fill_diagonal_(0.0)
    return rdm


def upper_triangle_vector(M: torch.Tensor, include_diagonal: bool = True) -> torch.Tensor:
    """Return the upper-triangular values of square matrix M as a 1D tensor."""
    N = M.size(0)
    offset = 0 if include_diagonal else 1
    idx = torch.triu_indices(N, N, offset=offset)
    return M[idx[0], idx[1]].to(dtype=torch.float32, device="cpu")