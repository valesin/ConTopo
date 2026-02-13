"""
Ensemble utilities: creation, hashing, saving/loading, and index management.
No diversity logic here.
"""

# Usage:
#   Import this module in your ensembling scripts to:
#     - Combine model outputs (logits/probs) using various methods
#     - Save/load ensemble results and metadata
#     - Manage ensemble index and hashes
#
#   Inference File Structure:
#   The inference results can be saved as a PyTorch file (e.g., `inference_cifar.pt`)
#   inside the model's run directory. This file can be a dictionary with:
#     - "preds": torch.Tensor (N,), predicted class indices.
#     - "labels": torch.Tensor (N,), ground truth labels.
#     - "logits": torch.Tensor (N, C), raw logits.
#     - "accuracy": float, model accuracy.
#   Or just a raw logits tensor.
#
#   Ensemble Cache Structure:
#   When an ensemble is created and saved, two files are generated in the `save_dir` with a unique hash:
#   1. `ensemble_{hash}.pt`: A PyTorch file containing a dictionary of the ensemble results.
#      - Keys: Ensemble method names (e.g., "soft", "hard").
#      - Values: The resulting probability or prediction tensors (torch.Tensor).
#   2. `ensemble_{hash}.json`: A JSON file containing metadata.
#      - "hash": The unique 16-char SHA-256 hash of the ensemble.
#      - "run_names": A sorted list of the model run names included in the ensemble.
#      - "metadata": Any additional metadata provided during creation.
#
#   Additionally, an `index.json` file in the `save_dir` is updated to map hashes to run names:
#   - Key: The ensemble hash.
#   - Value: The list of run names.
#
#   CLI Usage:
#     python utils/ensemble_utils.py --model_dirs dir1 dir2 --method soft --save_dir save/ensembles
#     python utils/ensemble_utils.py --config my_ensembles.yaml

import os
import json
import hashlib
import argparse
import yaml
import torch
from typing import List, Dict, Any, Optional

# -------------------------------
# Hashing and Index Management
# -------------------------------

def get_ensemble_hash(run_names: List[str]) -> str:
    """Deterministic 16-char SHA-256 hash of sorted run names."""
    return hashlib.sha256(json.dumps(sorted(run_names)).encode()).hexdigest()[:16]

def update_index(save_dir: str, run_hash: str, run_names: List[str]) -> None:
    """Update index.json with hash → run_names mapping."""
    index_path = os.path.join(save_dir, "index.json")
    index = {}
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
    index[run_hash] = sorted(run_names)
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

def load_index(save_dir: str) -> Dict[str, List[str]]:
    """Load the ensemble index file."""
    index_path = os.path.join(save_dir, "index.json")
    if not os.path.exists(index_path):
        return {}
    with open(index_path, "r") as f:
        return json.load(f)

# -------------------------------
# Ensemble Creation and I/O
# -------------------------------

def combine_logits(logits_list: List[torch.Tensor], method: str = "soft") -> torch.Tensor:
    """
    Combine logits from multiple models using the specified method.
    Supported: "soft" (average probs), "hard" (majority vote), etc.
    """
    M = len(logits_list)
    logits_stack = torch.stack(logits_list)  # [M, N, C]
    N, C = logits_stack.shape[1], logits_stack.shape[2]
    probs = torch.softmax(logits_stack, dim=2)  # [M, N, C]

    if method == "soft":
        return probs.mean(dim=0)  # [N, C]
    elif method == "hard":
        per_model_preds = logits_stack.argmax(dim=2)  # [M, N]
        hard_preds = torch.zeros(N, dtype=torch.long)
        for i in range(N):
            votes = per_model_preds[:, i]
            counts = torch.bincount(votes, minlength=C)
            hard_preds[i] = counts.argmax()
        hard_onehot = torch.zeros(N, C)
        hard_onehot.scatter_(1, hard_preds.unsqueeze(1), 1.0)
        return hard_onehot
    elif method == "max_confidence":
        max_conf_per_model = probs.max(dim=2).values  # [M, N]
        best_model_idx = max_conf_per_model.argmax(dim=0)  # [N]
        idx_expanded = best_model_idx.unsqueeze(0).unsqueeze(2).expand(1, N, C)  # [1, N, C]
        return probs.gather(0, idx_expanded).squeeze(0)  # [N, C]
    elif method == "conf_weighted":
        confs = probs.max(dim=2).values  # [M, N]
        weights = confs / confs.sum(dim=0, keepdim=True)  # [M, N]
        return torch.einsum("mn,mnc->nc", weights, probs)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

def save_ensemble(
    save_dir: str,
    run_names: List[str],
    ensemble_outputs: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save ensemble outputs and metadata. Returns the hash.
    """
    run_hash = get_ensemble_hash(run_names)
    os.makedirs(save_dir, exist_ok=True)
    pt_path = os.path.join(save_dir, f"ensemble_{run_hash}.pt")
    json_path = os.path.join(save_dir, f"ensemble_{run_hash}.json")

    torch.save(ensemble_outputs, pt_path)
    meta = {
        "hash": run_hash,
        "run_names": sorted(run_names),
        "metadata": metadata or {},
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    update_index(save_dir, run_hash, run_names)
    return run_hash

def load_ensemble(save_dir: str, run_hash: str) -> Dict[str, Any]:
    """Load ensemble outputs by hash."""
    pt_path = os.path.join(save_dir, f"ensemble_{run_hash}.pt")
    return torch.load(pt_path)

def list_ensembles(save_dir: str) -> List[str]:
    """List all ensemble hashes in the save directory."""
    index = load_index(save_dir)
    return list(index.keys())

# -------------------------------
# CLI Helpers
# -------------------------------

MODELS_ROOT = "save/ResNet18/models"

def get_trials(model_dir, trials):
    """Return list of trial names to use."""
    trial_path = os.path.join(MODELS_ROOT, model_dir)
    if not os.path.exists(trial_path):
         raise FileNotFoundError(f"Model directory not found: {trial_path}")
         
    all_trials = [d for d in os.listdir(trial_path) if os.path.isdir(os.path.join(trial_path, d)) and d.startswith("trial_")]
    if trials == "all":
        return sorted(all_trials)
    return [t for t in trials if t in all_trials]

def load_inference_from_path(path: str) -> torch.Tensor:
    """
    Load inference logits from a file path.

    Handles both raw logits tensors and dictionary outputs.
    If the file contains a dictionary, it expects keys like:
      - "preds": torch.Tensor (N,)
      - "labels": torch.Tensor (N,)
      - "logits": torch.Tensor (N, C)
      - "accuracy": float
    Values are extracted via the "logits" key.
    """
    if not os.path.exists(path):
         raise FileNotFoundError(f"Inference file not found: {path}")
    
    # Use weights_only=False to support legacy/complex saved data if safe globals aren't set
    # (Consistent with run_inference.py usage)
    data = torch.load(path, weights_only=False)
    
    if isinstance(data, dict) and "logits" in data:
        return data["logits"]
    return data

def load_inference_results(model_dir, trial, inference_file="inference.pt"):
    """Load inference results (e.g., logits.pt) from model_dir/trial."""
    path = os.path.join(MODELS_ROOT, model_dir, trial, inference_file)
    return load_inference_from_path(path)

# -------------------------------
# CLI Main
# -------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build and save ensembles from model directories or YAML config.",
        epilog="Example: python utils/ensemble_utils.py --model_dirs dir1 dir2 --method soft --save_dir save/ensembles\n         or: python utils/ensemble_utils.py --config my_ensembles.yaml"
    )
    parser.add_argument('--model_dirs', nargs='+', help='Directories containing inference file (one per model)')
    parser.add_argument('--inference_file', type=str, default='logits.pt', help='Filename of inference file in each directory (default: logits.pt)')
    parser.add_argument('--method', type=str, default='soft', help='Ensemble method (soft, hard, max_confidence, conf_weighted)')
    parser.add_argument('--save_dir', type=str, default='save/ensembles', help='Directory to save ensemble outputs')
    parser.add_argument('--metadata', type=str, default=None, help='Optional metadata as JSON string')
    parser.add_argument('--config', type=str, help='Path to YAML config file for batch ensemble creation')
    args = parser.parse_args()

    if args.config:
        # Fail if any other flag except --config is used
        forbidden = [
            ('model_dirs', args.model_dirs),
            ('inference_file', args.inference_file != 'logits.pt'),
            ('method', args.method != 'soft'),
            ('save_dir', args.save_dir != 'save/ensembles'),
            ('metadata', args.metadata is not None)
        ]
        used = [name for name, val in forbidden if val]
        if used:
            raise SystemExit(f"Error: When using --config, no other flags can be set (got: {', '.join(used)})")
        
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            
        for ens in config.get('ensembles', []):
            model_entries = ens.get('models', [])
            method = ens.get('method', 'soft')
            save_dir = ens.get('save_dir', 'save/ensembles')
            metadata = ens.get('metadata', {})

            run_names = []
            logits_list = []
            for m in model_entries:
                model_dir = m['name']
                trials = m.get('trials', 'all')
                trial_names = get_trials(model_dir, trials)
                for trial in trial_names:
                    run_name = f"{model_dir}___{trial}"
                    run_names.append(run_name)
                    # Use 'inference_file' from config, default to 'logits.pt' if not present
                    inf_file = ens.get('inference_file', 'logits.pt') 
                    logits = load_inference_results(model_dir, trial, inf_file)
                    logits_list.append(logits)

            if not logits_list:
                print(f"Warning: No models found for ensemble in {save_dir}. Skipping.")
                continue

            ensemble_probs = combine_logits(logits_list, method=method)
            ensemble_outputs = {method: ensemble_probs}
            run_hash = save_ensemble(save_dir, run_names, ensemble_outputs, metadata)
            print(f"Ensemble saved with hash: {run_hash}")
            print(f"Files: {save_dir}/ensemble_{run_hash}.pt, {save_dir}/ensemble_{run_hash}.json")
        return

    if args.model_dirs:
        run_names = [os.path.basename(os.path.normpath(d)) for d in args.model_dirs]
        logits_list = []
        for d in args.model_dirs:
            logits_path = os.path.join(d, args.inference_file)
            logits_list.append(load_inference_from_path(logits_path))

        ensemble_probs = combine_logits(logits_list, method=args.method)
        ensemble_outputs = {args.method: ensemble_probs}
        metadata = json.loads(args.metadata) if args.metadata else {}
        run_hash = save_ensemble(args.save_dir, run_names, ensemble_outputs, metadata)
        print(f"Ensemble saved with hash: {run_hash}")
        print(f"Files: {args.save_dir}/ensemble_{run_hash}.pt, {args.save_dir}/ensemble_{run_hash}.json")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()