import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from exp_errorcorr import _collect_errors_and_preds, _run_name, ensemble_accuracy

def infer_and_save(
    bundle,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    force: bool = False
) -> Dict[str, Any]:
    """
    Run inference on a single model bundle and save results to its run folder.
    
    This function generates a PyTorch file named `inference_cifar.pt` in the 
    same directory as the model checkpoint (bundle.meta["run_folder"]).
    
    The saved file contains a dictionary with the following keys:
      - "preds": torch.Tensor of shape (N,), the predicted class indices.
      - "labels": torch.Tensor of shape (N,), the ground truth labels.
      - "logits": torch.Tensor of shape (N, C), the raw logits.
      - "accuracy": float, the accuracy of this model on the test set.
    
    Args:
        bundle: LoadedModelBundle containing encoder, classifier, and meta.
        loader: DataLoader for evaluation.
        device: Device to run inference on.
        force: If True, re-run inference even if cache exists.
        
    Returns:
        Dict containing the same keys as the saved file.
    """
    run_folder = bundle.meta["run_folder"]
    cache_path = os.path.join(run_folder, "inference_cifar.pt")
    
    if not force and os.path.exists(cache_path):
        print(f"Loading cached inference from {cache_path}")
        return torch.load(cache_path, weights_only=False)
        
    # Ensure deterministic ordering:
    # If the loader shuffles, we cannot guarantee the output order matches the
    # indices or other cached files.
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
    
    accuracy = float((preds == labels).float().mean().item())
    
    results = {
        "preds": preds,
        "labels": labels, 
        "logits": logits,
        "accuracy": accuracy
    }
    
    torch.save(results, cache_path)
    print(f"Saved inference results to {cache_path}")
    
    return results

def evaluate_bundles_individually(
    bundles,
    loader: torch.utils.data.DataLoader,
    force: bool = False
) -> Dict[str, Any]:
    """
    Run inference for all bundles individually (cached per-trial) and aggregate results.
    
    Returns a dictionary structure compatible with exp_diversity.py expectations:
        - run_names: List[str]
        - accuracies: List[float]
        - error_matrix: torch.Tensor (num_trials, num_samples)
        - logits_matrix: torch.Tensor (num_trials, num_samples, num_classes)
        - labels_ref: torch.Tensor (num_samples)
        - param_vecs: List[torch.Tensor] (optional, if params analysis is added later)
        - ensemble_results: Dict[str, float]
        - non_ensemble_mean: float
        - non_ensemble_std: float
    """
    results_list = []
    run_names = []
    
    for bundle in bundles:
        # Determine device from bundle's encoder
        device = next(bundle.encoder.parameters()).device
        
        res = infer_and_save(bundle, loader, device, force=force)
        results_list.append(res)
        if hasattr(bundle, "run_name"):
            run_names.append(bundle.run_name)
        else:
            # Fallback: construct parent___trial to ensure uniqueness
            # This matches the _build_run_name logic in exp_diversity.py
            rf = Path(bundle.meta["run_folder"])
            run_names.append(f"{rf.parent.name}___{rf.name}")
        
    if not results_list:
        return None
        
    # Aggregate results
    # Check consistency of labels
    labels_ref = results_list[0]["labels"]
    for i, res in enumerate(results_list[1:]):
        if not torch.equal(res["labels"], labels_ref):
             raise RuntimeError(f"Mismatched label ordering in trial {run_names[i+1]}.")

    preds_all = [res["preds"] for res in results_list]
    logits_all = [res["logits"] for res in results_list]
    accuracies = [res["accuracy"] for res in results_list]
    
    # Reconstruct errors matrix (0 for correct, 1 for wrong)
    # shape: (num_trials, num_samples)
    errors_all = []
    for preds in preds_all:
        errors = (preds != labels_ref).float()
        errors_all.append(errors)
        
    error_matrix = torch.stack(errors_all)
    logits_matrix = torch.stack(logits_all)
    
    # Calculate non-ensemble stats
    non_ensemble_mean = float(np.mean(accuracies))
    non_ensemble_std = float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0
    
    # Calculate ensemble results
    ensemble_results = {}
    if logits_all:
        for method in ("soft", "hard", "max_confidence", "conf_weighted"):
            ensemble_results[method] = ensemble_accuracy(logits_all, labels_ref, method=method)

    # Collect params if needed (not strictly strictly required by prompt but good for compatibility)
    # The original _evaluate_bundles didn't return param_vecs unless explicitly added? 
    # Wait, the original _evaluate_bundles in exp_errorcorr DOES NOT return param_vecs. 
    # But exp_diversity.py main() checks for result.get("param_vecs") if do_params is True.
    # Ah, exp_errorcorr._evaluate_bundles does NOT collect params. 
    # So exp_diversity.py probably expected it to be added or it was missing in the file I viewed?
    # Let's check exp_diversity.py again. It says:
    # 311:         param_vecs = result.get("param_vecs")
    # 312:         if param_vecs is None:
    # 313:             print("No parameter vectors found.")
    # So it supports it being missing. I will ignore param_vecs for now as it wasn't in _evaluate_bundles either.
    
    return {
        "run_names": run_names,
        "accuracies": accuracies,
        "error_matrix": error_matrix,
        "logits_matrix": logits_matrix,
        "labels_ref": labels_ref,
        "ensemble_results": ensemble_results,
        "non_ensemble_mean": non_ensemble_mean,
        "non_ensemble_std": non_ensemble_std,
        # "param_vecs": ... (optional)
    }
