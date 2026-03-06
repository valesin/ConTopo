"""
Utility for running inference on model bundles and caching the results.

This module provides functions to run inference on loaded model bundles, calculate
accuracy, and save the resulting predictions, labels, and logits to a file for
efficient caching and subsequent analysis.

Saved File Structure:
The inference results are saved as a PyTorch file named `inference_cifar.pt` inside
the model's run directory. This file contains a dictionary with the following keys:
  - "preds": torch.Tensor of shape (N,), containing the predicted class indices for each sample.
  - "labels": torch.Tensor of shape (N,), containing the ground truth labels.
  - "logits": torch.Tensor of shape (N, C), containing the raw logits, where C is the number of classes.
  - "accuracy": float, representing the accuracy of the model on the evaluation dataset.
"""
import os
import torch
from typing import Dict, Any
from pathlib import Path
from torchvision import datasets, transforms
from utils.load import load_model_bundles
from utils.experiments import CIFAR10_MEAN, CIFAR10_STD
from utils import env

def run_model_inference(
    bundle,
    loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """
    Run inference on a single model bundle.
    
    Args:
        bundle: LoadedModelBundle containing encoder, classifier, and meta.
        loader: DataLoader for evaluation.
        device: Device to run inference on.
        
    Returns:
        Dict containing the same keys as the saved file.
    """
    
    # Ensure deterministic ordering:
    # If the loader shuffles, we cannot guarantee the output order matches the
    # indices or other cached files.ence_data to ensure deterministic ordering.
    if isinstance(loader.batch_sampler, torch.utils.data.BatchSampler):
        sampler = loader.batch_sampler.sampler
        if isinstance(sampler, torch.utils.data.RandomSampler):
            raise ValueError("Loader must use a deterministic sampler (shuffle=False) for inference caching.")
    elif isinstance(loader.sampler, torch.utils.data.RandomSampler):
         raise ValueError("Loader must use a deterministic sampler (shuffle=False) for inference caching.")

    print(f"Running inference for {_run_name(bundle.meta)}...")
    
    # _collect_errors_and_preds returns (errors, preds, targets, logits)
    # We discard errors as requested
    _, preds, labels, logits = _collect_errors_and_preds(
        bundle.encoder, 
        bundle.classifier, 
        loader, 
        device
    )

    # Compute embeddings for each test image using encoder only, in batches
    encoder = bundle.encoder
    encoder.eval()
    embeddings_list = []
    with torch.no_grad():
        for batch in loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device)
            feats = encoder(images)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            embeddings_list.append(feats.detach().cpu())
    embeddings = torch.cat(embeddings_list, dim=0)

    accuracy = float((preds == labels).float().mean().item())

    return {
        "preds": preds,
        "labels": labels, 
        "logits": logits,
        "accuracy": accuracy,
        "embeddings": embeddings
    }

# Module-level cache for the CIFAR-10 test loader
_CIFAR_LOADER = None

def _get_cifar_test_loader():
    """
    Get (and cache) a deterministic CIFAR-10 test DataLoader.
    
    Returns:
        torch.utils.data.DataLoader: Evaluation loader (shuffle=False).
    """
    global _CIFAR_LOADER
    if _CIFAR_LOADER is not None:
        return _CIFAR_LOADER
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    
    # Using defaults consistent with typical usage
    dataset = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
    _CIFAR_LOADER = torch.utils.data.DataLoader(
        dataset, 
        batch_size=256, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=torch.cuda.is_available()
    )
    return _CIFAR_LOADER

def get_or_run_inference(model_dir: str, trial: str, force: bool = False):
    """
    Get inference results for a specific model trial, running it if necessary.

    Args:
        model_dir: Directory relative to env.MODELS_ROOT (e.g. 'my_experiment')
        trial: Trial name (e.g. 'trial_00')
        force: If True, re-run inference even if cached.

    Returns:
        Dictionary containing standardized results:
        - "preds": torch.Tensor (N,)
        - "labels": torch.Tensor (N,)
        - "logits": torch.Tensor (N, C)
        - "accuracy": float
        - "embeddings": torch.Tensor (N, D)
    """
    # Construct full path to trial directory
    # env.MODELS_ROOT is defined in utils.ensemble_utils as e.g. "save/ResNet18/models"
    trial_dir = os.path.join(env.MODELS_ROOT, model_dir, trial)
    
    # Check cache
    cache_path = os.path.join(trial_dir, "inference_cifar.pt")
    
    required_keys = ["preds", "labels", "logits", "accuracy", "embeddings"]
    if os.path.exists(cache_path) and not force:
        data = torch.load(cache_path, weights_only=False)
        missing = False
        for k in required_keys:
            if k not in data or data[k] is None:
                missing = True
                break
        if not missing:
            return {
                "preds": data["preds"],
                "labels": data["labels"],
                "logits": data["logits"],
                "accuracy": data["accuracy"],
                "embeddings": data["embeddings"]
            }
        else:
            print(f"Cached inference for {model_dir}/{trial} is incomplete, re-running inference.")
        
    # Run Inference
    print(f"Running inference for {model_dir}/{trial}...")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model bundle
    # Note: load_model_bundles can handle loading logic. 
    # But it returns a list of bundles if passed a dir, usually.
    # If trial_dir is a run folder, it should return 1 bundle.
    bundles = load_model_bundles(
        trial_dir, 
        prefer="best", 
        device=device,
        eval_mode=True
    )
    
    if not bundles:
        raise FileNotFoundError(f"No model bundles found in {trial_dir}")
        
    bundle = bundles[0]
    
    # Get loader
    loader = _get_cifar_test_loader()
    
    # Run inference
    results = run_model_inference(bundle, loader, device)

    # Save results
    torch.save(results, cache_path)
    print(f"Saved inference results to {cache_path}")

    # Unified return for downstream use
    return {
        "preds": results["preds"],
        "labels": results["labels"],
        "logits": results["logits"],
        "accuracy": results["accuracy"],
        "embeddings": results["embeddings"]
    }

# ----- FROM exp_errorcorr.py -----
def _collect_errors_and_preds(
    encoder: torch.nn.Module,
    classifier: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Allocate per-sample buffers up front so batches can write into them without concatenation.
    total = len(loader.dataset)
    errors = torch.zeros(total, dtype=torch.float32)
    preds = torch.empty(total, dtype=torch.long)
    targets = torch.empty(total, dtype=torch.long)
    offset = 0
    logits_store: torch.Tensor | None = None

    encoder.eval()
    classifier.eval()

    with torch.no_grad():
        for batch in loader:
            # Some collates add extra metadata; slice to the canonical (images, labels).
            images, labels = batch[:2] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            feats = encoder(images)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            logits = classifier(feats)
            if isinstance(logits, (tuple, list)):
                logits = logits[-1]

            batch_preds = logits.argmax(dim=1)
            batch_errors = (batch_preds != labels).float().cpu()

            size = batch_errors.numel()
            errors[offset : offset + size] = batch_errors
            preds[offset : offset + size] = batch_preds.cpu()
            targets[offset : offset + size] = labels.cpu()
            logits_cpu = logits.detach().cpu()
            if logits_store is None:
                # Lazily size the logits tensor once we know how many classes the head emits.
                logits_store = torch.empty(total, logits_cpu.size(1), dtype=logits_cpu.dtype)
            logits_store[offset : offset + size] = logits_cpu
            offset += size

    if logits_store is None:
        logits_store = torch.empty(total, 0)

    return errors, preds, targets, logits_store

def _run_name(meta: dict) -> str:
    run_folder = meta.get("run_folder")
    if run_folder:
        return Path(run_folder).name
    ckpt = meta.get("ckpt_path")
    if ckpt:
        return Path(ckpt).parent.name
    return "run"

def get_cifar10_test_labels():
    """
    Returns the ground truth labels for the CIFAR-10 test set in the same order as used in inference.
    """
    from torchvision import datasets, transforms
    from utils.experiments import CIFAR10_MEAN, CIFAR10_STD
    import torch

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    dataset = datasets.CIFAR10(root=env.DATA_ROOT, train=False, download=True, transform=transform)
    labels = torch.tensor(dataset.targets)
    return labels

# Function to retrieve all inference results for a model directory
def get_inference_bundles(model_dir):
    """
    Returns a list of inference results for all trials in a model directory.
    Each result is a dict with keys: preds, labels, logits, accuracy, embeddings.
    """
    import os
    import torch
    from utils import env

    model_root = os.path.join(env.MODELS_ROOT, model_dir)
    if not os.path.isdir(model_root):
        raise FileNotFoundError(f"Model directory not found: {model_root}")
    trial_dirs = [d for d in os.listdir(model_root) if d.startswith("trial_")]
    trial_dirs.sort()  # Ensure consistent order
    bundles = []
    for trial in trial_dirs:
        cache_path = os.path.join(model_root, trial, "inference_cifar.pt")
        if os.path.exists(cache_path):
            data = torch.load(cache_path, weights_only=False)
            bundles.append(data)
    return bundles

# Function to retrieve all titles of inference files for a model directory
def get_inference_title_bundles(model_dir):
    """
    Returns a list of inference file paths for all trials in a model directory.
    Each result is a string path to an inference file.
    """
    import os
    from utils import env

    model_root = os.path.join(env.MODELS_ROOT, model_dir)
    if not os.path.isdir(model_root):
        raise FileNotFoundError(f"Model directory not found: {model_root}")
    trial_dirs = [d for d in os.listdir(model_root) if d.startswith("trial_")]
    trial_dirs.sort()  # Ensure consistent order
    bundles = []
    for trial in trial_dirs:
        cache_path = os.path.join(model_root, trial, "inference_cifar.pt")
        if os.path.exists(cache_path):
            bundles.append(cache_path)
    return bundles