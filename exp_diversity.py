"""
Analyze Diversity: Compute and store ensemble diversity metrics.

Always computes all requested metrics and overwrites output files,
so that diversity.csv and pairwise_diversity.pt always reflect the
current run config exactly.

Usage:
    python exp_diversity.py --config configs/diversity.yaml

Config Format (YAML):
    metrics:
      - pred_disagreement
      - q_statistic
      # ...
    save_dir: env.ENSEMBLES_ROOT       # Optional, default env.ENSEMBLES_ROOT
    ensembles:                       # Optional, if missing/empty, runs on ALL ensembles in save_dir
      - "hash1"
      - "hash2"
"""

import argparse
import yaml
import os
import csv

import torch
import numpy as np
from typing import List, Dict, Any, Tuple

# Project imports
import utils.metrics as metrics_lib
from utils.ensemble_utils import (
    load_registry,
    resolve_identifier,
    get_ensemble_path_by_name
)
from utils.names import parse_run_name
from utils.run_inference import get_or_run_inference
from configs import env

def _load_diversity_csv(csv_path: str) -> Dict[str, float]:
    """Load scalar diversity metrics from CSV."""
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

def _save_diversity_csv(csv_path: str, metrics: Dict[str, float]):
    """Write scalar diversity metrics to CSV, replacing any existing file."""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])

def _load_pairwise_pt(pt_path: str) -> Dict[str, torch.Tensor]:
    """Load pairwise metrics from .pt file."""
    if not os.path.exists(pt_path):
        return {}
    try:
        return torch.load(pt_path, map_location='cpu')
    except Exception as e:
        print(f"  Warning: Could not load {pt_path}: {e}")
        return {}

def get_diversity_results(run_hash: str, save_dir: str = env.ENSEMBLES_ROOT) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
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
    
    scalars = _load_diversity_csv(div_csv_path)
    pairwise = _load_pairwise_pt(pair_pt_path)
    
    return scalars, pairwise

def get_diversity_results_by_ensemble_name(ensemble_name: str, save_dir: str = env.ENSEMBLES_ROOT) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    """
    Retrieve diversity and pairwise metrics for a given ensemble name (not hash).
    Uses get_ensemble_path_by_name to resolve the directory.
    Args:
        ensemble_name: The name of the ensemble (not the hash).
        save_dir: The directory where ensembles are saved.
    Returns:
        tuple: (scalar_diversity_dict, pairwise_diversity_dict)
    """
    hash_dir = get_ensemble_path_by_name(ensemble_name, save_dir)
    div_csv_path = os.path.join(hash_dir, "diversity.csv")
    pair_pt_path = os.path.join(hash_dir, "pairwise_diversity.pt")
    scalars = _load_diversity_csv(div_csv_path)
    pairwise = _load_pairwise_pt(pair_pt_path)
    return scalars, pairwise

def compute_metrics(
    logits_list: List[torch.Tensor],
    labels: torch.Tensor,
    metric_names: List[str]
) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    """
    Compute specified diversity metrics (scalar and pairwise).
    Uses the registry-based metrics API via EvalContext.
    """
    preds_list = [logits.argmax(dim=1) for logits in logits_list]

    ctx = metrics_lib.EvalContext(
        preds=preds_list,
        labels=labels,
        logits=logits_list,
    )

    # Scalar results (off-diagonal average)
    scalar_results = metrics_lib.compute_metrics(ctx, metric_names, reduce_group=True)

    # Pairwise results (full R×R matrices)
    pairwise_raw = metrics_lib.compute_metrics(ctx, metric_names, reduce_group=False)
    pairwise_results = {
        k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
        for k, v in pairwise_raw.items()
    }

    return scalar_results, pairwise_results

def process_ensemble(
    run_hash: str,
    run_names: List[str],
    save_dir: str,
    required_metrics: List[str]
):
    """
    Process a single ensemble: compute all requested metrics and save.
    Always overwrites existing output files.
    """
    hash_dir = os.path.join(save_dir, run_hash)
    os.makedirs(hash_dir, exist_ok=True)
    
    div_csv_path = os.path.join(hash_dir, "diversity.csv")
    pair_pt_path = os.path.join(hash_dir, "pairwise_diversity.pt")

    # 1. Load data (logits/labels) for each member
    logits_list = []
    labels = None
    
    for run_name in run_names:
        try:
            model_dir, trial = parse_run_name(run_name)
            inf_data = get_or_run_inference(model_dir, trial)
            logits_list.append(inf_data['logits'])
            
            if labels is None:
                labels = inf_data['labels']
            elif not torch.equal(labels, inf_data['labels']):
                print(f"    Warning: Label mismatch for {run_name}")
                
        except Exception as e:
            print(f"    Error loading {run_name}: {e}")
            print(f"    Skipping {run_hash}: Data loading failed.")
            return
            
    if not logits_list:
        print(f"    Skipping {run_hash}: No data loaded.")
        return

    # 2. Compute all requested metrics
    scalars, pairwise = compute_metrics(logits_list, labels, required_metrics)
    
    # 3. Save (overwrite)
    _save_diversity_csv(div_csv_path, scalars)
    torch.save(pairwise, pair_pt_path)
        
    print(f"    Saved: {list(scalars.keys())}")


def main():
    parser = argparse.ArgumentParser(description="Diversity Analysis")
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
    
    # Load Registry
    registry = load_registry(save_dir)
    if not registry:
        print(f"No registry found or empty in {save_dir}. Ensure ensembles are created first.")
        return

    # Determine which hashes to process (supports names and hashes)
    if not target_hashes:
        print("No specific ensembles listed in config. Processing ALL known ensembles.")
        target_hashes = list(registry.keys())
    else:
        # Resolve names/hashes to validated hashes
        valid_hashes = []
        for identifier in target_hashes:
            try:
                h = resolve_identifier(identifier, save_dir)
                valid_hashes.append(h)
            except KeyError:
                print(f"Warning: '{identifier}' not found in registry.")
        target_hashes = valid_hashes

    print(f"Processing {len(target_hashes)} ensembles...")

    for i, run_hash in enumerate(target_hashes):
        entry = registry[run_hash]
        run_names = entry["run_names"]
        display = entry.get("name") or run_hash
        print(f"[{i+1}/{len(target_hashes)}] Ensemble {display} ({len(run_names)} models)")
        process_ensemble(run_hash, run_names, save_dir, metrics)
        
    print("Done.")

if __name__ == "__main__":
    main()
