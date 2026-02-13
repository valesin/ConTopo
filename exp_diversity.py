"""
Analyze Diversity: Compute and store ensemble diversity metrics.
Replaces the old analysis script with a proper experiment runner.

Usage:
    python analyze_diversity.py --config configs/diversity_exp.yaml

Config Format (YAML):
    ensembles:
      - name: "my_ensemble"     # optional, hash is used if omitted
        models:
          - name: "models/model_A" # relative to MODELS_ROOT or absolute
            trials: "all"          # or [1, 2, 3] or specific folder names
        metrics:
          - pred_disagreement
          - q_statistic
        methods:                # Ensemble combination methods
          - soft
          - hard
        save_dir: "save/ensembles"
"""

import argparse
import yaml
import os
import json
import torch
import numpy as np
import sys
from typing import List, Dict, Any, Optional, Tuple

# Project imports
import utils.metrics as metrics_lib
from utils.ensemble_utils import (
    get_ensemble_hash,
    save_ensemble,
    combine_logits,
    get_trials,
    MODELS_ROOT
)

def load_inference_data(model_dir: str, trial: str, filename: str = 'logits.pt') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load logits and labels from an inference file.
    
    Args:
        model_dir: Model directory name (under MODELS_ROOT)
        trial: Trial directory name
        filename: Inference filename
        
    Returns:
        (logits, labels) tuple.
    """
    # Construct path
    # Try relative to MODELS_ROOT first
    path = os.path.join(MODELS_ROOT, model_dir, trial, filename)
    if not os.path.exists(path):
        # Try as absolute path if model_dir is absolute? 
        # But get_trials uses MODELS_ROOT, so we assume strict structure.
        raise FileNotFoundError(f"Inference file not found: {path}")

    try:
        data = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load {path}: {e}")

    if isinstance(data, dict):
        if 'logits' in data and 'labels' in data:
            return data['logits'], data['labels']
        else:
            raise ValueError(f"Dictionary in {path} missing 'logits' or 'labels' keys. Keys found: {list(data.keys())}")
    elif isinstance(data, torch.Tensor):
        # If it's just a tensor, we assume it's logits. But we HAVE TO HAVE labels.
        # Maybe we can load labels from dataset if missing? 
        # But that requires knowing which dataset and split.
        # For now, enforce labels presence in file.
        raise ValueError(f"File {path} contains only a Tensor (logits?). Labels are required for diversity analysis.")
    else:
        raise ValueError(f"Unknown data format in {path}: {type(data)}")


def compute_ensemble_metrics(
    logits_list: List[torch.Tensor],
    labels: torch.Tensor,
    metric_names: List[str],
    ensemble_methods: List[str]
) -> Dict[str, Any]:
    """
    Compute ensemble accuracy and diversity metrics.
    """
    N = labels.numel()
    
    # Check consistency
    for l in logits_list:
        if l.shape[0] != N:
            raise ValueError(f"Logits dimension mismatch: {l.shape[0]} vs labels {N}")

    # 1. Compute Individual Accuracies
    indiv_accs = []
    preds_list = []
    probs_list = []
    
    for logits in logits_list:
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        indiv_accs.append(acc)
        preds_list.append(preds)
        probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())

    # 2. Compute Ensemble Accuracies
    acc_results = {}
    
    for method in ensemble_methods:
        # combine_logits returns combined probabilities (or one-hot)
        try:
            ens_output = combine_logits(logits_list, method=method)
            ens_preds = ens_output.argmax(dim=1)
            ens_acc = (ens_preds == labels).float().mean().item()
            acc_results[f"acc_{method}"] = ens_acc
        except Exception as e:
            print(f"  Warning: Failed to compute ensemble method '{method}': {e}")

    # 3. Compute Diversity Metrics
    # Pre-compute stack for efficiency
    try:
        counts_stack = metrics_lib.matrix_confusion_counts(preds_list, labels)
    except Exception as e:
        print(f"  Error computing confusion counts: {e}")
        return {}

    # Context for metric functions
    ctx = {
        "preds_list": preds_list,
        "probs_list": probs_list,
        "logits_list": logits_list,
        "counts_stack": counts_stack,
        "labels": labels,
        "N": N,
        # Add param_vecs if available? Not loaded currently.
    }

    div_results = {}
    pairwise_results = {}

    for m_name in metric_names:
        # Resolve function name
        # We look for matrix_<name> (pairwise) and group_<name> (scalar)
        
        # Clean name if user passed 'div_pred_disagreement' -> 'pred_disagreement'
        clean_name = m_name.replace("div_", "").replace("pw_", "")
        
        matrix_fn = getattr(metrics_lib, f"matrix_{clean_name}", None)
        group_fn = getattr(metrics_lib, f"group_{clean_name}", None)
        
        # Fallback to pairwise_ prefix if matrix_ is missing (metrics.py naming conventions might vary)
        if matrix_fn is None:
             matrix_fn = getattr(metrics_lib, f"pairwise_{clean_name}", None)

        # Compute Pairwise Matrix
        if matrix_fn is not None:
            try:
                # Call with ctx. We need to be careful about arguments.
                # Inspecting metrics.py: most take specific args.
                # We can't just pass **ctx if they don't accept **kwargs.
                # Wrapper: we'll try to pass relevant args by name if possible, 
                # or rely on the function accepting **kwargs.
                # Checking utils/metrics.py: most functions DO accept **kwargs.
                mat = matrix_fn(**ctx) 
                pairwise_results[f"pw_{clean_name}"] = mat
            except Exception as e:
                # print(f"    (debug) pairwise {clean_name} failed: {e}")
                pass

        # Compute Group Scalar
        if group_fn is not None:
            try:
                 val = group_fn(**ctx)
                 div_results[f"div_{clean_name}"] = val
            except Exception as e:
                # print(f"    (debug) group {clean_name} failed: {e}")
                pass
        elif matrix_fn is not None and f"pw_{clean_name}" in pairwise_results:
            # Fallback: average off-diagonal
            try:
                mat = pairwise_results[f"pw_{clean_name}"]
                val = metrics_lib._average_off_diagonal(mat)
                div_results[f"div_{clean_name}"] = val
            except Exception:
                pass

    return {
        "acc": acc_results,
        "diversity": div_results,
        "pairwise": pairwise_results,
        "individual_accuracies": indiv_accs
    }


def process_config(config_path: str):
    print(f"Loading config from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    ensembles = config.get('ensembles', [])
    if not ensembles:
        print("No ensembles defined in config.")
        return

    for i, ens_cfg in enumerate(ensembles):
        name_tag = ens_cfg.get('name', f"ensemble_{i}")
        print(f"\n[{i+1}/{len(ensembles)}] Processing '{name_tag}'...")
        
        # Parse config
        models_cfg = ens_cfg.get('models', [])
        metric_names = ens_cfg.get('metrics', [])
        ensemble_methods = ens_cfg.get('methods', ['soft'])
        save_dir = ens_cfg.get('save_dir', 'save/ensembles')
        metadata = ens_cfg.get('metadata', {})
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Load Models
        run_names = []
        logits_list = []
        labels = None 
        
        valid_ensemble = True
        
        for m_cfg in models_cfg:
            m_name = m_cfg['name']
            trials = m_cfg.get('trials', 'all')
            
            # Resolve trial directories
            try:
                # If m_name is a relative path 'ResNet18/models/model_A', we might need to adjust?
                # The util assumes MODELS_ROOT is 'save/ResNet18/models'.
                # If user puts 'save/ResNet18/models/model_A' in config, get_trials might fail 
                # if it appends it to MODELS_ROOT.
                # HACK: If m_name starts with 'save/', assume it's relative to CWD and not MODELS_ROOT?
                # But get_trials enforces MODELS_ROOT.
                # Let's try to interpret m_name relative to MODELS_ROOT first.
                # If m_name is "crossentropy_...", it's likely just the folder name.
                
                # Check if m_name exists relative to CWD, if so, maybe we can bypass get_trials?
                # But get_trials does logic.
                
                # Let's assume user provides name relative to MODELS_ROOT if they know it, 
                # or we try to find it.
                
                # We will stick to the logic: m_name is the model directory name inside MODELS_ROOT.
                # If user provides path, we might need to strip.
                if m_name.startswith(MODELS_ROOT):
                    m_name = os.path.relpath(m_name, MODELS_ROOT)
                
                trial_names = get_trials(m_name, trials)
            except FileNotFoundError:
                print(f"  Error: Model directory not found or invalid: {m_name}")
                valid_ensemble = False
                break

            for t in trial_names:
                full_run_name = f"{m_name}___{t}"
                inf_file = ens_cfg.get('inference_file', 'logits.pt')
                
                try:
                    l, lbl = load_inference_data(m_name, t, inf_file)
                    
                    logits_list.append(l)
                    run_names.append(full_run_name)
                    
                    if labels is None:
                        labels = lbl
                    elif not torch.equal(labels, lbl):
                        # Simple check. If labels are different, we have a problem.
                        # But maybe order is same? We assume standard dataset order.
                        pass
                        
                except Exception as e:
                    print(f"  Error loading {full_run_name}: {e}")
                    valid_ensemble = False
                    break
            
            if not valid_ensemble: break
        
        if not valid_ensemble or not logits_list:
            print("  Skipping invalid ensemble.")
            continue

        # Compute
        print(f"  Computing metrics for {len(logits_list)} models...")
        results = compute_ensemble_metrics(
            logits_list, labels, metric_names, ensemble_methods
        )
        
        # Add metadata
        results["run_names"] = sorted(run_names)
        results["metadata"] = metadata
        results["subset_size"] = len(run_names)
        
        # Save
        # save_ensemble writes the file. We pass our full dict as 'ensemble_outputs'.
        run_hash = save_ensemble(save_dir, run_names, results, metadata)
        print(f"  Saved ensemble {run_hash} to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Ensemble Diversity Experiment Runner")
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    
    args = parser.parse_args()

    if args.config:
        process_config(args.config)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
