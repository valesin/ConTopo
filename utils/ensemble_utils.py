"""
Utility for building and saving ensembles from multiple model trials.

Example YAML config format:
--------------------------
ensembles:
  - ensemble_name: "my_ensemble" # Optional unique human-readable name
    models:
      - name: "model_dir_1"
        trials: ["trial_00", "trial_01"]
      - name: "model_dir_2"
        trials: "all"
    methods: ["soft", "hard"]  # Optional, defaults to all supported
    metadata:                  # Optional arbitrary user data
      description: "Example ensemble"
s
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
from datetime import datetime, timezone
from utils.names import generate_run_name, get_trials, parse_run_name
from utils.run_inference import get_or_run_inference
from typing import List, Dict, Any, Optional, Callable, Iterator
from utils import env
import numpy as np
from configs import env

# ---------------------------------------------------------------
# Similarity Profiles Iterator
# ---------------------------------------------------------------

def iter_similarity_profiles(save_dir: str = env.ENSEMBLES_ROOT):
    """
    Yields for each registered ensemble:
        {
            "ensemble_name": name or hash,
            "hash": hash,
            "similarity_profiles": dict from load_similarity_profiles(...)
        }
    Skips ensembles without similarity_profiles.pt.
    """
    # Use config file to determine ensemble order and selection
    config_path = os.path.join(env.CONFIGS_ROOT, "ensembles.yaml")
    import yaml
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Ensemble config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    for ens in config.get("ensembles", []):
        name = ens.get("ensemble_name")
        # Fallback to None if not present
        if name is None:
            continue
        try:
            profiles = load_similarity_profiles(name, save_dir=save_dir)
        except FileNotFoundError:
            continue
        # Optionally, get hash from registry for completeness
        try:
            hash_ = resolve_identifier(name, save_dir)
        except Exception:
            hash_ = None
        yield {
            "ensemble_name": name,
            "hash": hash_,
            "similarity_profiles": profiles
        }

def iter_ensemble_inference_data_from_config(config_path: str = os.path.join(env.CONFIGS_ROOT, "ensembles.yaml")) -> Iterator[Dict[str, Any]]:
    """
    Yields for each ensemble in the config file:
        {
            "ensemble_name": ...,
            "run_names": [...],
            "methods": [...],
            "metadata": {...},
            "inference_data": dict mapping run_name -> inference data
        }
    """
    import yaml
    from utils.names import get_trials, generate_run_name, parse_run_name
    from utils.run_inference import get_or_run_inference
    import os

    # Default config path: save/ensembles/../ensembles.yaml
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Ensemble config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    for ens in config.get("ensembles", []):
        model_entries = ens.get("models", [])
        run_names = []
        for m in model_entries:
            model_dir = m["name"]
            trials = m.get("trials", "all")
            trial_names = get_trials(model_dir, trials)
            for trial in trial_names:
                run_names.append(generate_run_name(model_dir, trial))

        def get_inference_data():
            inf_data_dict = {}
            for run_name in run_names:
                model_dir, trial = parse_run_name(run_name)
                inf_data = get_or_run_inference(model_dir, trial)
                inf_data_dict[run_name] = inf_data
            return inf_data_dict

        yield {
            "ensemble_name": ens.get("ensemble_name"),
            "run_names": run_names,
            "methods": ens.get("methods", []),
            "metadata": ens.get("metadata", {}),
            "inference_data": get_inference_data(),
        }


def get_ensemble_config_path_from_cli(default_path: str = 'configs/ensembles.yaml') -> str:
    """
    Parse CLI arguments for ensemble config path, defaulting to 'configs/ensembles.yaml'.
    Returns the config path as a string.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Ensemble experiment utility.")
    parser.add_argument('--config', type=str, default=default_path, help='Path to ensemble config YAML file')
    args, _ = parser.parse_known_args()
    return args.config

METHODS = ["soft", "hard", "max_confidence", "conf_weighted"]

# ---------------------------------------------------------------
# Registry Management
# ---------------------------------------------------------------

def _registry_path(save_dir: str = env.ENSEMBLES_ROOT) -> str:
    return os.path.join(save_dir, "registry.json")


def _index_path(save_dir: str = env.ENSEMBLES_ROOT) -> str:
    return os.path.join(save_dir, "index.json")


def load_registry(save_dir: str = env.ENSEMBLES_ROOT) -> Dict[str, Dict[str, Any]]:
    """
    Load the ensemble registry.

    Auto-migrates from legacy index.json if registry.json does not exist.
    Returns a dict keyed by ensemble hash, each value containing:
      name, run_names, methods, metadata, created, updated.
    """
    reg_path = _registry_path(save_dir)
    if os.path.exists(reg_path):
        with open(reg_path, "r") as f:
            return json.load(f)

    # Auto-migrate from legacy index.json
    idx_path = _index_path(save_dir)
    if os.path.exists(idx_path):
        return _migrate_index_to_registry(save_dir)

    return {}


def save_registry(registry: Dict[str, Dict[str, Any]], save_dir: str = env.ENSEMBLES_ROOT) -> None:
    """Write the full registry to disk."""
    os.makedirs(save_dir, exist_ok=True)
    reg_path = _registry_path(save_dir)
    with open(reg_path, "w") as f:
        json.dump(registry, f, indent=2)


def update_registry(
    run_hash: str,
    run_names: List[str],
    methods: List[str],
    save_dir: str = env.ENSEMBLES_ROOT,
    metadata: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
) -> None:
    """
    Create or overwrite a single registry entry.

    Enforces name uniqueness: raises ValueError if `name` is already used
    by a different hash.
    """
    registry = load_registry(save_dir)
    now = datetime.now(timezone.utc).isoformat()

    # Enforce name uniqueness
    if name:
        for h, entry in registry.items():
            if entry.get("name") == name and h != run_hash:
                raise ValueError(
                    f"Name '{name}' is already used by hash {h}. "
                    f"Names must be unique."
                )

    created = registry.get(run_hash, {}).get("created", now)
    registry[run_hash] = {
        "name": name,
        "run_names": sorted(run_names),
        "methods": sorted(methods),
        "metadata": metadata or {},
        "created": created,
        "updated": now,
    }
    save_registry(registry, save_dir)


def _migrate_index_to_registry(save_dir: str = env.ENSEMBLES_ROOT) -> Dict[str, Dict[str, Any]]:
    """Migrate legacy index.json to registry.json, reading per-dir metadata."""
    idx_path = _index_path(save_dir)
    with open(idx_path, "r") as f:
        index = json.load(f)

    now = datetime.now(timezone.utc).isoformat()
    registry: Dict[str, Dict[str, Any]] = {}

    for run_hash, run_names in index.items():
        # Try to read per-directory metadata
        meta_path = os.path.join(save_dir, run_hash, "metadata.json")
        metadata = {}
        name = None
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                full_meta = json.load(f)
            metadata = full_meta.get("metadata", {})
            # Use metadata.name as fallback name
            name = metadata.pop("name", None)

        # Detect which methods have .pt files
        hash_dir = os.path.join(save_dir, run_hash)
        methods = []
        if os.path.isdir(hash_dir):
            for fname in os.listdir(hash_dir):
                stem = fname.replace(".pt", "")
                if fname.endswith(".pt") and stem in METHODS:
                    methods.append(stem)

        registry[run_hash] = {
            "name": name,
            "run_names": sorted(run_names),
            "methods": sorted(methods),
            "metadata": metadata,
            "created": now,
            "updated": now,
        }

    save_registry(registry, save_dir)
    print(f"Migrated {len(registry)} entries from index.json → registry.json")
    return registry


# ---------------------------------------------------------------
# Identifier Resolution
# ---------------------------------------------------------------

def resolve_identifier(identifier: str, save_dir: str = env.ENSEMBLES_ROOT) -> str:
    """
    Resolve a name or hash to a registry hash.

    Accepts either a hash (exact match) or a name (looked up in registry).
    Raises KeyError if not found.
    """
    registry = load_registry(save_dir)

    # Direct hash match
    if identifier in registry:
        return identifier

    # Search by name
    for h, entry in registry.items():
        if entry.get("name") == identifier:
            return h

    raise KeyError(f"No ensemble found for identifier '{identifier}'")


def _get_registry_entry(identifier: str, save_dir: str = env.ENSEMBLES_ROOT) -> tuple:
    """Helper: resolve identifier and return (hash, entry) from registry."""
    run_hash = resolve_identifier(identifier, save_dir)
    registry = load_registry(save_dir)
    return run_hash, registry[run_hash]


# ---------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------

def get_ensemble_hash(run_names: List[str]) -> str:
    """Deterministic 16-char SHA-256 hash of sorted run names."""
    return hashlib.sha256(
        json.dumps(sorted(run_names)).encode()
    ).hexdigest()[:16]


# ---------------------------------------------------------------
# Ensemble Creation and I/O
# ---------------------------------------------------------------

def combine_logits(logits_list: List[torch.Tensor], method: str = "soft") -> torch.Tensor:
    """
    Combine logits from multiple models using the specified method.

    Supported methods:
      - "soft": average class probabilities
      - "hard": majority vote (returns one-hot)
      - "max_confidence": pick model with highest confidence per sample
      - "conf_weighted": confidence-weighted average of probabilities
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
        idx_expanded = best_model_idx.unsqueeze(0).unsqueeze(2).expand(1, N, C)
        return probs.gather(0, idx_expanded).squeeze(0)  # [N, C]
    elif method == "conf_weighted":
        confs = probs.max(dim=2).values  # [M, N]
        weights = confs / confs.sum(dim=0, keepdim=True)  # [M, N]
        return torch.einsum("mn,mnc->nc", weights, probs)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")


def save_ensemble(
    run_names: List[str],
    ensemble_probs: torch.Tensor,
    method: str,
    save_dir: str = env.ENSEMBLES_ROOT,
    metadata: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
) -> str:
    """
    Save ensemble outputs, per-dir metadata, and update registry.

    Creates save_dir/{hash}/ with {method}.pt and metadata.json.
    Returns the hash.
    """
    run_hash = get_ensemble_hash(run_names)
    hash_dir = os.path.join(save_dir, run_hash)
    os.makedirs(hash_dir, exist_ok=True)

    # Save prediction probabilities
    pt_path = os.path.join(hash_dir, f"{method}.pt")
    torch.save(ensemble_probs, pt_path)

    return run_hash


def _save_per_dir_metadata(
    run_hash: str,
    run_names: List[str],
    methods: List[str],
    save_dir: str = env.ENSEMBLES_ROOT,
    metadata: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
) -> None:
    """Write per-directory metadata.json (kept for portability)."""
    hash_dir = os.path.join(save_dir, run_hash)
    os.makedirs(hash_dir, exist_ok=True)
    meta = {
        "hash": run_hash,
        "name": name,
        "run_names": sorted(run_names),
        "methods": sorted(methods),
        "metadata": metadata or {},
    }
    json_path = os.path.join(hash_dir, "metadata.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)


def load_ensemble(identifier: str, save_dir: str = env.ENSEMBLES_ROOT) -> Dict[str, torch.Tensor]:
    """
    Load ensemble outputs by name or hash.

    Returns a dict keyed by method name -> probability tensor.
    """
    run_hash = resolve_identifier(identifier, save_dir)
    hash_dir = os.path.join(save_dir, run_hash)
    result = {}
    for fname in os.listdir(hash_dir):
        stem = fname.replace(".pt", "")
        if fname.endswith(".pt") and stem in METHODS:
            result[stem] = torch.load(
                os.path.join(hash_dir, fname), weights_only=False
            )
    return result


def load_metadata(identifier: str, save_dir: str = env.ENSEMBLES_ROOT) -> Dict[str, Any]:
    """
    Load ensemble custom metadata by name or hash.

    Returns the custom metadata dict (e.g., {"rho": 0}).
    """
    _, entry = _get_registry_entry(identifier, save_dir)
    return entry.get("metadata", {})


# ---------------------------------------------------------------
# Listing and Selection
# ---------------------------------------------------------------

def list_ensembles(save_dir: str = env.ENSEMBLES_ROOT) -> List[Dict[str, Any]]:
    """
    List all registered ensembles, sorted deterministically by name (or hash).
    Returns a list of dicts, each with 'hash' and 'name' keys.
    """
    registry = load_registry(save_dir)
    ensembles = [
        {"hash": h, "name": entry.get("name")}
        for h, entry in registry.items()
    ]
    # Sort by name, fallback to hash if name is None
    return sorted(ensembles, key=lambda x: (x["name"] or x["hash"]))


def select_ensembles(
    save_dir: str = env.ENSEMBLES_ROOT,
    names: Optional[List[str]] = None,
    filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> List[Dict[str, Any]]:
    """
    Select ensembles by name list and/or metadata filter.

    Returns full info dicts (same shape as get_ensemble_info output)
    for each matching ensemble. If both names and filter_fn are given,
    both conditions must be satisfied.

    Examples:
        # By names
        select_ensembles(save_dir, names=["CE_rho0", "CE_rho0.2"])

        # By metadata filter
        select_ensembles(save_dir, filter_fn=lambda m: m.get("rho", 1) < 0.1)

        # Combined
        select_ensembles(save_dir, names=[...], filter_fn=lambda m: ...)
    """
    registry = load_registry(save_dir)
    results = []

    # Build name set for fast lookup
    name_set = set(names) if names else None

    for run_hash, entry in registry.items():
        # Name filter
        if name_set is not None:
            entry_name = entry.get("name")
            if entry_name not in name_set and run_hash not in name_set:
                continue

        # Metadata filter
        if filter_fn is not None:
            try:
                if not filter_fn(entry.get("metadata", {})):
                    continue
            except Exception:
                continue

        results.append(get_ensemble_info(run_hash, save_dir))

    return results


# ---------------------------------------------------------------
# Accuracy Utilities
# ---------------------------------------------------------------

def get_ensemble_accuracy(ensemble_probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute accuracy of ensemble predictions against true labels."""
    ensemble_preds = ensemble_probs.argmax(dim=1)
    correct = (ensemble_preds == labels).sum().item()
    total = labels.size(0)
    return correct / total if total > 0 else 0.0


def get_ensemble_accuracies(
    ensemble_outputs: Dict[str, torch.Tensor], labels: torch.Tensor
) -> Dict[str, float]:
    """
    Given a dict of method_name -> probs (from load_ensemble),
    compute accuracy for each method.
    """
    return {
        method: get_ensemble_accuracy(probs, labels)
        for method, probs in ensemble_outputs.items()
    }


def get_component_accuracies(run_names: List[str]) -> Dict[str, Any]:
    """
    Compute accuracy of each component model from cached inference.

    Returns dict with mean_acc, max_acc, per_component (list), num_components.
    """
    comp_accs = []
    for rn in run_names:
        model_dir, trial = parse_run_name(rn)
        inf_data = get_or_run_inference(model_dir, trial)
        comp_accs.append(inf_data["accuracy"])
    comp_accs = np.array(comp_accs)
    return {
        "comp_mean_acc": float(comp_accs.mean()),
        "comp_max_acc": float(comp_accs.max()),
        "comp_accs": comp_accs.tolist(),
        "num_components": len(run_names),
    }


# ---------------------------------------------------------------
# High-level Utility Functions
# ---------------------------------------------------------------

def load_similarity_profiles(identifier: str, save_dir: str = env.ENSEMBLES_ROOT) -> dict:
    """
    Retrieve similarity profiles (cosine_results, rdm_mats) for an ensemble by name or hash.
    Returns a dict with keys 'cosine_results' and 'rdm_mats'.
    """
    run_hash = resolve_identifier(identifier, save_dir)
    ensemble_dir = os.path.join(save_dir, run_hash)
    similarity_file = os.path.join(ensemble_dir, "similarity_profiles.pt")
    if not os.path.isfile(similarity_file):
        raise FileNotFoundError(f"No similarity_profiles.pt found for ensemble '{identifier}' at {similarity_file}")
    return torch.load(similarity_file, weights_only=False)

def get_ensemble_info(identifier: str, save_dir: str = env.ENSEMBLES_ROOT) -> Dict[str, Any]:
    """
    Get full information about an ensemble by name or hash.

    Returns a dict with:
      - hash, name, run_names, num_components, methods, metadata
      - comp_mean_acc, comp_max_acc, comp_accs (from cached inference)
      - ensemble_accs: {method: accuracy} for each computed method
    """
    run_hash, entry = _get_registry_entry(identifier, save_dir)
    run_names = entry["run_names"]

    # Component accuracies (from cached inference)
    comp = get_component_accuracies(run_names)

    # Ensemble accuracies (load .pt files + labels)
    ensemble_outputs = load_ensemble(run_hash, save_dir)
    ensemble_accs = {}
    if ensemble_outputs:
        # Get labels from any component's cached inference
        model_dir, trial = parse_run_name(run_names[0])
        labels = get_or_run_inference(model_dir, trial)["labels"]
        ensemble_accs = get_ensemble_accuracies(ensemble_outputs, labels)

    return {
        "hash": run_hash,
        "name": entry.get("name"),
        "run_names": run_names,
        "num_components": comp["num_components"],
        "methods": entry.get("methods", []),
        "metadata": entry.get("metadata", {}),
        "comp_mean_acc": comp["comp_mean_acc"],
        "comp_max_acc": comp["comp_max_acc"],
        "comp_accs": comp["comp_accs"],
        "ensemble_accs": ensemble_accs,
    }


def get_ensemble_accuracy_by_id(
    identifier: str, method: str, save_dir: str = env.ENSEMBLES_ROOT
) -> float:
    """Get ensemble accuracy for a specific method, by name or hash."""
    run_hash, entry = _get_registry_entry(identifier, save_dir)
    run_names = entry["run_names"]

    hash_dir = os.path.join(save_dir, run_hash)
    pt_path = os.path.join(hash_dir, f"{method}.pt")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(
            f"Method '{method}' not found for ensemble '{identifier}'"
        )

    probs = torch.load(pt_path, weights_only=False)
    model_dir, trial = parse_run_name(run_names[0])
    labels = get_or_run_inference(model_dir, trial)["labels"]
    return get_ensemble_accuracy(probs, labels)


def get_component_stats(identifier: str, save_dir: str = env.ENSEMBLES_ROOT) -> Dict[str, Any]:
    """
    Get component-level accuracy stats for an ensemble, by name or hash.

    Returns: {mean_acc, max_acc, num_components, per_component}.
    """
    _, entry = _get_registry_entry(identifier, save_dir)
    comp = get_component_accuracies(entry["run_names"])
    return {
        "mean_acc": comp["comp_mean_acc"],
        "max_acc": comp["comp_max_acc"],
        "num_components": comp["num_components"],
        "per_component": comp["comp_accs"],
    }


def get_num_components(identifier: str, save_dir: str = env.ENSEMBLES_ROOT) -> int:
    """Get the number of components in an ensemble, by name or hash."""
    _, entry = _get_registry_entry(identifier, save_dir)
    return len(entry["run_names"])

def get_ensemble_path_by_name(name: str, save_dir: str = env.ENSEMBLES_ROOT) -> str:
    """
    Return the full path to the ensemble directory given its name or hash.
    Raises KeyError if not found.
    """
    run_hash = resolve_identifier(name, save_dir)
    return os.path.join(save_dir, run_hash)



# ---------------------------------------------------------------
# CLI Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build and save ensembles from YAML config.",
        epilog="Example: python utils/ensemble_utils.py --config my_ensembles.yaml",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file for batch ensemble creation",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    for ens in config.get("ensembles", []):
        model_entries = ens.get("models", [])
        methods = ens.get("methods", METHODS.copy())
        save_dir = ens.get("save_dir", env.ENSEMBLES_ROOT)
        metadata = ens.get("metadata", {})

        # Resolve ensemble name: top-level 'ensemble_name' takes priority,
        # fall back to metadata.name for backward compat
        name = ens.get("ensemble_name") or metadata.pop("name", None)

        # Collect run names
        run_names = []
        for m in model_entries:
            model_dir = m["name"]
            trials = m.get("trials", "all")
            trial_names = get_trials(model_dir, trials)
            for trial in trial_names:
                run_names.append(generate_run_name(model_dir, trial))

        if not run_names:
            print(f"Warning: No models found for ensemble '{name}'. Skipping.")
            continue

        run_hash = get_ensemble_hash(run_names)
        print(f"Ensemble '{name or run_hash}' ({len(run_names)} models, hash {run_hash})")

        # Always load logits and recompute all methods
        logits_list = []
        for run_name in run_names:
            model_dir, trial = parse_run_name(run_name)
            inf_data = get_or_run_inference(model_dir, trial)
            logits_list.append(inf_data["logits"])

        for method in methods:
            ensemble_probs = combine_logits(logits_list, method=method)
            save_ensemble(run_names, ensemble_probs, method, save_dir=save_dir)
            print(f"  Saved {method}.pt")

        # Overwrite per-directory metadata and registry entry
        _save_per_dir_metadata(run_hash, run_names, methods, save_dir=save_dir, metadata=metadata, name=name)
        update_registry(run_hash, run_names, methods, save_dir=save_dir, metadata=metadata, name=name)
        print(f"  Registry and metadata updated.")


if __name__ == "__main__":
    main()