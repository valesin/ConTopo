"""
Analyze Diversity: Compute and store ensemble diversity metrics.
Supports incremental calculation and pairwise metric saving.

Usage:
    python exp_diversity.py --config configs/diversity.yaml

Config Format (YAML):
    metrics:
      - pred_disagreement
      - q_statistic
      # ...
    save_dir: "save/ensembles"       # Optional, default "save/ensembles"
    ensembles:                       # Optional, if missing/empty, runs on ALL ensembles in save_dir
      - "hash1"
      - "hash2"
"""

import argparse
import yaml
import os
import csv
import json
import inspect
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set

# Project imports
import utils.metrics as metrics_lib
from utils.ensemble_utils import (
    load_index,
    parse_run_name,
    list_ensembles
)
from utils.env import MODELS_ROOT
from utils.run_inference import get_or_run_inference

def load_diversity_csv(csv_path: str) -> Dict[str, float]:
    """Load existing scalar diversity metrics from CSV."""
    if not os.path.exists(csv_path):
        return {}
    
    metrics = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    try:
                        metrics[row[0]] = float(row[1])
                    except ValueError:
                        pass # Skip header or invalid
    except Exception as e:
        print(f"  Warning: Could not read {csv_path}: {e}")
    return metrics

def save_diversity_csv(csv_path: str, new_metrics: Dict[str, float]):
    """Append new metrics to diversity.csv."""
    # We append to avoid overwriting if other processes are checking (though not concurrency safe)
    # Better to read, merge, write for consistency, but append is redundant-safe if we check first.
    # Given the requirement "check diversity.csv... if exists but contains few metrics",
    # we should probably append only missing ones.
    
    # Actually, simpler to just append the new variable lines. 
    # Duplicate keys in CSV are messy but readable. Ideally we invoke this only for missing ones.
    
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["metric", "value"])
        for k, v in new_metrics.items():
            writer.writerow([k, v])

def load_pairwise_pt(pt_path: str) -> Dict[str, torch.Tensor]:
    """Load existing pairwise metrics."""
    if not os.path.exists(pt_path):
        return {}
    try:
        return torch.load(pt_path, map_location='cpu')
    except Exception as e:
        print(f"  Warning: Could not load {pt_path}: {e}")
        return {}

def get_diversity_results(run_hash: str, save_dir: str = "save/ensembles") -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    """
    Retrieve diversity and pairwise metrics for a given ensemble hash.
    
    Args:
        run_hash: The hash of the ensemble.
        save_dir: The directory where ensembles are saved.
        
    Returns:
        tuple: (scalar_diversity_dict, pairwise_diversity_dict)
    """
    hash_dir = os.path.join(save_dir, run_hash)
    div_csv_path = os.path.join(hash_dir, "diversity.csv")
    pair_pt_path = os.path.join(hash_dir, "pairwise_diversity.pt")
    
    scalars = load_diversity_csv(div_csv_path)
    pairwise = load_pairwise_pt(pair_pt_path)
    
    return scalars, pairwise

def compute_metrics(
    logits_list: List[torch.Tensor],
    labels: torch.Tensor,
    metric_names: List[str]
) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    """
    Compute specified diversity metrics (scalar and pairwise).
    """
    N = labels.numel()
    
    # Pre-compute useful structures
    preds_list = []
    probs_list = []
    for logits in logits_list:
        preds_list.append(logits.argmax(dim=1))
        # probs_list might be heavy if many models/samples, but needed for some metrics
        probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
    
    # Pre-compute confusion counts stack (used by many metrics)
    try:
        counts_stack = metrics_lib.matrix_confusion_counts(preds_list, labels)
    except Exception as e:
        print(f"  Error computing confusion counts: {e}")
        return {}, {}

    ctx = {
        "preds_list": preds_list,
        "probs_list": probs_list,
        "logits_list": logits_list,
        "counts_stack": counts_stack,
        "labels": labels,
        "N": N
    }

    scalar_results = {}
    pairwise_results = {}

    for m_name in metric_names:
        # We try to compute both group (scalar) and matrix (pairwise) versions for each requested metric
        # Check for function existence in utils.metrics
        
        # Clean name: remove prefixes if user provided them, though standard config should just be 'jaccard' etc.
        base_name = m_name.replace("div_", "").replace("pw_", "").replace("group_", "").replace("matrix_", "")
        
        matrix_fn = getattr(metrics_lib, f"matrix_{base_name}", None)
        group_fn = getattr(metrics_lib, f"group_{base_name}", None)
        
        # Calculate Pairwise
        if matrix_fn:
            try:
                sig = inspect.signature(matrix_fn)
                kwargs = {k: ctx[k] for k in sig.parameters if k in ctx}
                # Check for top_n or specific args if needed, currently generic handler
                mat_np = matrix_fn(**kwargs)
                pairwise_results[base_name] = torch.from_numpy(mat_np)
            except Exception as e:
                print(f"    Error computing pairwise {base_name}: {e}")

        # Calculate Scalar
        if group_fn:
            try:
                sig = inspect.signature(group_fn)
                kwargs = {k: ctx[k] for k in sig.parameters if k in ctx}
                val = group_fn(**kwargs)
                scalar_results[base_name] = float(val)
            except Exception as e:
                print(f"    Error computing scalar {base_name}: {e}")
        
        # Fallback: if scalar not computed but pairwise exists, avg off-diagonal
        if base_name not in scalar_results and base_name in pairwise_results:
             mat = pairwise_results[base_name].numpy()
             val = metrics_lib._average_off_diagonal(mat)
             if not np.isnan(val):
                 scalar_results[base_name] = float(val)

    return scalar_results, pairwise_results

def process_ensemble(
    run_hash: str,
    run_names: List[str],
    save_dir: str,
    required_metrics: List[str]
):
    """
    Process a single ensemble: check existing results, compute missing, save.
    """
    hash_dir = os.path.join(save_dir, run_hash)
    os.makedirs(hash_dir, exist_ok=True)
    
    div_csv_path = os.path.join(hash_dir, "diversity.csv")
    pair_pt_path = os.path.join(hash_dir, "pairwise_diversity.pt")
    
    # 1. Check existing work
    existing_scalars = load_diversity_csv(div_csv_path)
    existing_pairwise = load_pairwise_pt(pair_pt_path)
    
    missing_metrics = list(set(required_metrics) - 
        (set(existing_scalars.keys()) & set(existing_pairwise.keys()))
    )
    
    if not missing_metrics:
        print(f"  [{run_hash}] All metrics up to date.")
        return

    print(f"  [{run_hash}] Computing missing: {missing_metrics}")

    # 2. Load Data (Logits/Labels)
    # We only load data if we have work to do
    logits_list = []
    labels = None
    
    valid_ensemble = True
    for run_name in run_names:
        try:
            model_dir, trial = parse_run_name(run_name)
            inf_data = get_or_run_inference(model_dir, trial)
            l = inf_data['logits']
            lbl = inf_data['labels']
            logits_list.append(l)
            
            if labels is None:
                labels = lbl
            elif not torch.equal(labels, lbl):
                pass # Warning could be logged
                
        except Exception as e:
            print(f"    Error loading {run_name}: {e}")
            valid_ensemble = False
            break
            
    if not valid_ensemble or not logits_list:
        print(f"    Skipping {run_hash}: Data loading failed.")
        return

    # 3. Compute
    new_scalars, new_pairwise = compute_metrics(logits_list, labels, missing_metrics)
    
    # 4. Save
    if new_scalars:
        save_diversity_csv(div_csv_path, new_scalars)
    
    if new_pairwise:
        # Merge with existing
        existing_pairwise.update(new_pairwise)
        torch.save(existing_pairwise, pair_pt_path)
        
    print(f"    Saved: {list(new_scalars.keys())}")


def main():
    parser = argparse.ArgumentParser(description="Diversity Analysis (Incremental)")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        return

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    metrics = config.get('metrics', [])
    if not metrics:
        print("No 'metrics' found in config.")
        return
        
    save_dir = config.get('save_dir', 'save/ensembles')
    target_hashes = config.get('ensembles', [])

    print(f"Output Directory: {save_dir}")
    
    # Load Main Index
    index = load_index(save_dir)
    if not index:
        print(f"No index found or empty in {save_dir}. Ensure ensembles are created first.")
        return

    # Determine which hashes to process
    if not target_hashes:
        print("No specific ensembles listed in config. Processing ALL known ensembles.")
        target_hashes = list(index.keys())
    else:
        # Validate requested hashes exist
        valid_hashes = []
        for h in target_hashes:
            if h in index:
                valid_hashes.append(h)
            else:
                print(f"Warning: Requested hash {h} not found in index.")
        target_hashes = valid_hashes

    print(f"Processing {len(target_hashes)} ensembles...")

    for i, run_hash in enumerate(target_hashes):
        run_names = index[run_hash]
        print(f"[{i+1}/{len(target_hashes)}] Ensemble {run_hash} ({len(run_names)} models)")
        process_ensemble(run_hash, run_names, save_dir, metrics)
        
    print("Done.")

if __name__ == "__main__":
    main()
