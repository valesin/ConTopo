"""
Ensemble combination utilities.

Provides functions to compute ensemble probabilities from a set of models,
supporting multiple combination strategies (soft, hard, max_confidence, conf_weighted).
Also handles caching of ensemble results with traceability back to specific trials.

Hashing helpers (``get_ensemble_hash``, ``update_index``) provide deterministic
mapping between a sorted list of run names and a 16-hex-char SHA-256 hash used
as the filesystem key for cached ensemble results.
"""

from __future__ import annotations

import hashlib
import json
import os
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union

import torch


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def get_ensemble_hash(run_names: List[str]) -> str:
    """
    Generate a deterministic 16-hex-char hash for a set of run names.

    The hash is the first 16 characters of the SHA-256 digest of the
    JSON-encoded *sorted* list of run names.  Sorting guarantees that
    the same set of models always maps to the same hash regardless of
    insertion order.
    """
    return hashlib.sha256(json.dumps(sorted(run_names)).encode()).hexdigest()[:16]


def update_index(save_dir: str, run_hash: str, run_names: List[str]) -> None:
    """
    Append / update an entry in the global ``index.json`` inside *save_dir*.

    ``index.json`` maps every previously computed ensemble hash to the
    sorted list of its component run names, enabling fast reverse lookups.
    """
    index_path = os.path.join(save_dir, "index.json")
    index: Dict[str, List[str]] = {}
    if os.path.exists(index_path):
        try:
            with open(index_path, "r") as f:
                index = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    index[run_hash] = sorted(run_names)
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)


# ---------------------------------------------------------------------------
# Ensemble Strategies
# ---------------------------------------------------------------------------


def ensemble_probs_for_subset(logits_subset: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute ensemble probability tensors for all 4 methods from a subset of model logits.

    Args:
        logits_subset: shape [M, N, C] — logits from M models over N samples and C classes.

    Returns:
        Dict mapping method name -> ensemble probability tensor of shape [N, C].
        For 'hard' voting the tensor is a one-hot encoding of the majority-vote prediction.
    """
    M, N, C = logits_subset.shape
    probs = torch.softmax(logits_subset, dim=2)  # [M, N, C]

    results: Dict[str, torch.Tensor] = {}

    # --- soft: average probabilities ---
    results["soft"] = probs.mean(dim=0)  # [N, C]

    # --- hard: majority vote (one-hot) ---
    per_model_preds = logits_subset.argmax(dim=2)  # [M, N]
    hard_preds = torch.zeros(N, dtype=torch.long)
    for i in range(N):
        votes = per_model_preds[:, i]
        counts = torch.bincount(votes, minlength=C)
        hard_preds[i] = counts.argmax()
    hard_onehot = torch.zeros(N, C)
    hard_onehot.scatter_(1, hard_preds.unsqueeze(1), 1.0)
    results["hard"] = hard_onehot

    # --- max_confidence: per sample, use the full prob vector from the most confident model ---
    max_conf_per_model = probs.max(dim=2).values  # [M, N]
    best_model_idx = max_conf_per_model.argmax(dim=0)  # [N]
    # Gather the probability vectors from the best model for each sample
    # best_model_idx: [N] -> expand to [1, N, C] for gather over dim 0
    idx_expanded = best_model_idx.unsqueeze(0).unsqueeze(2).expand(1, N, C)  # [1, N, C]
    results["max_confidence"] = probs.gather(0, idx_expanded).squeeze(0)  # [N, C]

    # --- conf_weighted: weight each model's probs by its confidence ---
    confs = max_conf_per_model  # [M, N]
    weights = confs / confs.sum(dim=0, keepdim=True)  # [M, N], normalised
    # einsum: (M,N) * (M,N,C) -> (N,C)
    results["conf_weighted"] = torch.einsum("mn,mnc->nc", weights, probs)

    return results


# ---------------------------------------------------------------------------
# Caching & Computation
# ---------------------------------------------------------------------------


def compute_all_ensemble_probs(
    logits_matrix: torch.Tensor,
    labels: torch.Tensor,
    run_names: List[str],
    cache_dir: Optional[str] = None,
) -> Dict[Union[str, Tuple[int, int]], Dict[str, torch.Tensor]]:
    """
    Compute ensemble probability tensors for all 4 methods, both for the full
    ensemble (all R models) and for every pairwise combination (i, j) where i < j.

    Args:
        logits_matrix: shape [R, N, C].
        labels: shape [N] (ground-truth labels; not used in computation but
                kept for potential downstream use).
        run_names: List of run names corresponding to the R models.
                   Used for cache filename generation.
        cache_dir: Directory to search for/save cache. Actual cache will be in
                   a generic 'ensembles' subdirectory within this path.

    Returns:
        Dict keyed by "all" or (i, j) tuples.  Each value is a dict mapping
        method name ("soft", "hard", "max_confidence", "conf_weighted") to a
        torch.Tensor of shape [N, C].
    """
    R = logits_matrix.shape[0]
    if len(run_names) != R:
        raise ValueError(
            f"Number of run_names ({len(run_names)}) does not match logits dimension ({R})"
        )

    # 1. Determine cache path
    cache_path = None
    meta_path = None
    
    if cache_dir is not None:
        ensembles_dir = os.path.join(cache_dir, "ensembles")
        os.makedirs(ensembles_dir, exist_ok=True)
        
        # Consistent sorting for hashing
        sorted_runs = sorted(run_names)
        run_hash = hashlib.sha256(json.dumps(sorted_runs).encode()).hexdigest()[:16]
        
        cache_filename = f"ensemble_{run_hash}.pt"
        meta_filename = f"ensemble_{run_hash}.json"
        
        cache_path = os.path.join(ensembles_dir, cache_filename)
        meta_path = os.path.join(ensembles_dir, meta_filename)

        # 2. Try loading
        if os.path.exists(cache_path) and os.path.exists(meta_path):
            print(f"Loading cached ensemble probs from {cache_path}")
            # Optional: verify metadata matches run_names
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                if set(meta.get("run_names", [])) == set(run_names):
                     return torch.load(cache_path, weights_only=False)
                else:
                    print("Warning: Hash collision or metadata mismatch. Recomputing.")
            except Exception as e:
                print(f"Error loading metadata: {e}. Recomputing.")

    # 3. Compute
    result: Dict[Union[str, Tuple[int, int]], Dict[str, torch.Tensor]] = {}

    # Full ensemble (all R models)
    result["all"] = ensemble_probs_for_subset(logits_matrix)

    # Pairwise ensembles
    for i, j in combinations(range(R), 2):
        result[(i, j)] = ensemble_probs_for_subset(logits_matrix[[i, j]])

    # 4. Save
    if cache_path is not None and meta_path is not None:
        print(f"Saving ensemble probs to cache at {cache_path}")
        torch.save(result, cache_path)
        with open(meta_path, "w") as f:
            json.dump({"run_names": run_names}, f, indent=2)

    return result


def get_trials_from_cache(cache_path_or_name: str) -> List[str]:
    """
    Given a cache path (e.g. .../ensemble_abc123.pt) or filename,
    retrieve the list of run names involved from the sidecar JSON.
    """
    base, ext = os.path.splitext(cache_path_or_name)
    if ext != ".pt":
        # If user passed json path, just use it
        if ext == ".json":
            meta_path = cache_path_or_name
        else:
             # Assume it's a name without extension or weird extension
             meta_path = base + ".json"
    else:
        meta_path = base + ".json"
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        
    with open(meta_path, "r") as f:
        data = json.load(f)
        
    return data.get("run_names", [])
