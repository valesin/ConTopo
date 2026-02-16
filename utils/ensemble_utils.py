"""
Utility for building and saving ensembles from multiple model trials.

Example YAML config format:
--------------------------
ensembles:
  - models:
      - name: "model_dir_1"
        trials: ["trial_00", "trial_01"]
      - name: "model_dir_2"
        trials: "all"
    methods: ["soft", "hard"]  # Optional, defaults to all supported
    save_dir: "save/ensembles" # Optional, defaults to 'save/ensembles'
    metadata:                  # Optional
      description: "Example ensemble"

Usage:
------
python utils/ensemble_utils.py --config configs/ensembles.yaml
OR
python -m utils.ensemble_utils --config configs/ensembles.yaml
"""

import os
import json
import hashlib
import argparse
import yaml
import torch
from utils.run_inference import get_or_run_inference
from typing import List, Dict, Any, Optional, Tuple
from utils import env

METHODS = ["soft", "hard", "max_confidence", "conf_weighted"]
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
    ensemble_probs: torch.Tensor,
    method: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save ensemble outputs and metadata. Returns the hash.

    Creates `save_dir/{hash}/` with one `{method}.pt` per key in
    ensemble_outputs, plus a single `metadata.json`.
    """
    run_hash = get_ensemble_hash(run_names)
    hash_dir = os.path.join(save_dir, run_hash)
    os.makedirs(hash_dir, exist_ok=True)

    pt_path = os.path.join(hash_dir, f"{method}.pt")
    torch.save(ensemble_probs, pt_path)

    meta = {
        "hash": run_hash,
        "run_names": sorted(run_names),
        "metadata": metadata or {},
    }
    json_path = os.path.join(hash_dir, "metadata.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    update_index(save_dir, run_hash, run_names)
    return run_hash

def load_ensemble(save_dir: str, run_hash: str) -> Dict[str, Any]:
    """
    Load ensemble outputs by hash.

    Reads all `.pt` files in `save_dir/{hash}/` (excluding `diversity.pt`)
    and returns a dict keyed by method name.
    """
    hash_dir = os.path.join(save_dir, run_hash)
    result = {}
    for fname in os.listdir(hash_dir):
        if fname.endswith(".pt") and fname.replace(".pt", "") in METHODS:
            method_name = fname.replace(".pt", "")
            result[method_name] = torch.load(os.path.join(hash_dir, fname))
    return result

def load_metadata(save_dir: str, run_hash: str) -> Dict[str, Any]:
    """
    Load ensemble custom metadata by hash.
    
    Returns only the custom metadata dict from the config (e.g., {"rho": 0}).
    """
    metadata_path = os.path.join(save_dir, run_hash, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, "r") as f:
        full_meta = json.load(f)
        return full_meta.get("metadata", {})

def list_ensembles(save_dir: str) -> List[str]:
    """List all ensemble hashes in the save directory."""
    index = load_index(save_dir)
    return list(index.keys())

def get_trials(model_dir, trials):
    """Return list of trial names to use."""
    trial_path = os.path.join(env.MODELS_ROOT, model_dir)
    if not os.path.exists(trial_path):
         raise FileNotFoundError(f"Model directory not found: {trial_path}")
         
    all_trials = [d for d in os.listdir(trial_path) if os.path.isdir(os.path.join(trial_path, d)) and d.startswith("trial_")]
    if trials == "all":
        return sorted(all_trials)
    return [t for t in trials if t in all_trials]

def parse_run_name(run_name: str) -> Tuple[str, str]:
    """
    Parse the run name into model_dir and trial.
    Format is {model_dir}___{trial}.
    """
    if "___" not in run_name:
        raise ValueError(f"Invalid run_name format (missing separator '___'): {run_name}")
    
    # Split on the last occurrence of '___' as model_dir might contain it (though unlikely)
    parts = run_name.rsplit("___", 1)
    if len(parts) != 2:
        # Should be covered by the "___" check but for type safety
        raise ValueError(f"Could not parse run_name: {run_name}")
        
    model_dir, trial = parts
    return model_dir, trial

# -------------------------------
# CLI Main
# -------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build and save ensembles from YAML config.",
        epilog="Example: python utils/ensemble_utils.py --config my_ensembles.yaml"
    )
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file for batch ensemble creation')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    for ens in config.get('ensembles', []):
        model_entries = ens.get('models', [])
        methods = ens.get('methods', ['soft', 'hard', 'max_confidence', 'conf_weighted'])
        save_dir = ens.get('save_dir', 'save/ensembles')
        metadata = ens.get('metadata', {})

        run_names = []
        # First pass: collect run names to compute hash
        for m in model_entries:
            model_dir = m['name']
            trials = m.get('trials', 'all')
            trial_names = get_trials(model_dir, trials)
            for trial in trial_names:
                run_names.append(f"{model_dir}___{trial}")
        
        if not run_names:
            print(f"Warning: No models found for ensemble in {save_dir}. Skipping.")
            continue

        # Check which methods need to be computed
        run_hash = get_ensemble_hash(run_names)
        methods_to_run = []
        for method in methods:
            if not os.path.exists(os.path.join(save_dir, run_hash, f"{method}.pt")):
                methods_to_run.append(method)
            else:
                print(f"Method {method} already exists for hash {run_hash}. Skipping.")

        # Check if metadata needs updating
        metadata_path = os.path.join(save_dir, run_hash, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                saved_meta = json.load(f)
            if saved_meta.get("metadata") != metadata:
                print(f"Updating metadata for hash {run_hash}...")
                saved_meta["metadata"] = metadata
                with open(metadata_path, 'w') as f:
                    json.dump(saved_meta, f, indent=2)
        elif not methods_to_run and metadata: # Case where methods exist but metadata file doesn't (rare but possible)
             # Should probably create it if we have metadata to save
             print(f"Creating missing metadata file for hash {run_hash}...")
             meta = {
                "hash": run_hash,
                "run_names": sorted(run_names),
                "metadata": metadata or {},
            }
             with open(metadata_path, "w") as f:
                json.dump(meta, f, indent=2)

        
        if not methods_to_run:
            print(f"All methods for hash {run_hash} are already computed.")
            continue

        # Second pass: load logits only if needed
        logits_list = []
        for run_name in run_names:
            model_dir, trial = parse_run_name(run_name)
            logits_dict = get_or_run_inference(model_dir, trial)
            logits_list.append(logits_dict['logits'])

        for method in methods_to_run:  # ['soft', 'hard', 'max_confidence', 'conf_weighted']
            ensemble_probs = combine_logits(logits_list, method=method)
            run_hash = save_ensemble(save_dir, run_names, ensemble_probs, method, metadata)
            print(f"Ensemble saved with hash: {run_hash}")
            print(f"Files: {save_dir}/{run_hash}/{method}.pt")

if __name__ == "__main__":
    main()