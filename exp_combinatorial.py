"""
Combinatorial Diversity vs. Performance Analysis.

This script loads a set of models defined in a YAML config, iterates through all
possible subsets (ensembles) of size 2 to N, and computes:
1. Ensemble accuracy (soft, hard, etc.)
2. Diversity metrics (pairwise for pairs, group average for larger subsets)

CLI Usage
---------
::

    # Full run (all subsets)
    python exp_combinatorial.py --config configs/test_combinatorial.yaml

    # Quick test (limit to 5 random subsets per k)
    python exp_combinatorial.py --config configs/test_combinatorial.yaml --limit_subsets 5

    # Force CPU
    python exp_combinatorial.py --config configs/test_combinatorial.yaml --device cpu

YAML Config Fields
------------------
::

    model_dir: "save/ResNet18/models"       # root directory containing model folders
    runs:                                    # (optional) restrict to these model folders
      - "crossentropy_wstopo_torus_..."
    data_root: "dataset"                     # CIFAR-10 location
    batch_size: 100
    save_dir: "save/ensembles"               # where ensemble results are saved
    min_k: 2                                 # smallest subset size
    max_k: 3                                 # largest subset size (defaults to R)
    metrics:                                 # (optional) diversity metrics to compute
      - pred_disagreement                    #   omit to use all available
      - q_statistic
    ensemble_methods:                        # (optional) ensemble strategies
      - soft                                 #   omit to use all (soft/hard/max_confidence/conf_weighted)
      - hard

Output Files
------------
Each evaluated ensemble produces three things inside ``save_dir``:

- ``ensemble_<hash>.pt``  — torch dict with ``acc``, ``diversity``, ``run_names``,
  ``hash``, ``subset_size``, ``metadata``
- ``ensemble_<hash>.json`` — sidecar JSON with hash, run_names, subset_size,
  and lists of computed metrics
- ``index.json`` — global lookup mapping *every* hash to its component run names

Hash ↔ Ensemble Components
---------------------------
The hash is a deterministic SHA-256 (truncated to 16 hex chars) of the **sorted**
list of run names in the ensemble.

**From hash → component models** (lookup)::

    import json

    # Option 1: global index (fast, all ensembles)
    with open("save/ensembles/index.json") as f:
        index = json.load(f)
    components = index["273be8d129eb1dce"]  # list of run name strings

    # Option 2: per-ensemble sidecar JSON
    with open("save/ensembles/ensemble_273be8d129eb1dce.json") as f:
        meta = json.load(f)
    components = meta["run_names"]

    # Option 3: from the .pt file itself
    import torch
    data = torch.load("save/ensembles/ensemble_273be8d129eb1dce.pt")
    components = data["run_names"]       # list of strings
    accuracies = data["acc"]             # dict: "acc_soft" -> float, ...
    diversity  = data["diversity"]       # dict: "div_pred_disagreement" -> float, ...

**From model names → hash** (compute)::

    import hashlib, json
    run_names = ["modelA___trial_00", "modelB___trial_01"]
    run_hash = hashlib.sha256(json.dumps(sorted(run_names)).encode()).hexdigest()[:16]
    # Then load: torch.load(f"save/ensembles/ensemble_{run_hash}.pt")

**Iterate all ensembles**::

    import json, torch, os
    save_dir = "save/ensembles"
    with open(os.path.join(save_dir, "index.json")) as f:
        index = json.load(f)
    for h, names in index.items():
        data = torch.load(os.path.join(save_dir, f"ensemble_{h}.pt"))
        print(f"{h}: k={data['subset_size']}, acc_soft={data['acc']['acc_soft']:.4f}")
"""

import sys
import os
import argparse
import hashlib
import json
import yaml
from typing import List, Dict, Any, Tuple, Callable, Optional
from itertools import combinations

import torch
import numpy as np

# Add repo root to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.experiments import load_config, parse_model_name
from utils.ensemble_combination import ensemble_probs_for_subset
from utils.metrics import (
    pairwise_confusion_counts,
    pairwise_pred_disagreement,
    pairwise_q_statistic,
    pairwise_output_correlation,
    pairwise_double_fault,
    pairwise_jaccard,
    pairwise_cohens_kappa,
    pairwise_correctness_disagreement,
    pairwise_error_conditional_disagreement,
    pairwise_overall_agreement,
    pairwise_iou_top_n,
    group_pred_disagreement,
    group_q_statistic,
    group_output_correlation,
    group_double_fault,
    group_jaccard,
    group_cohens_kappa,
    group_correctness_disagreement,
    group_error_conditional_disagreement,
    group_overall_agreement,
    group_iou_top_n,
    _average_off_diagonal,
)
from utils.load import load_model_bundles
from utils.run_inference import evaluate_bundles_individually
from exp_errorcorr import _load_clean_test_images, _normalize_images
from utils.experiments import CIFAR10_MEAN, CIFAR10_STD
from torch.utils.data import TensorDataset, DataLoader


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

# Each entry maps a config name to:
#   (pairwise_fn, group_fn, extra_kwargs_builder)
#
# extra_kwargs_builder is a callable that receives a context dict and returns
# kwargs for the metric function, or None if no extras are needed.

def _pw_disagreement(ctx):
    """Pairwise prediction disagreement (no counts_stack needed)."""
    return pairwise_pred_disagreement(ctx["preds_list"])

def _gp_disagreement(ctx):
    return group_pred_disagreement(ctx["preds_list"])

def _pw_q_statistic(ctx):
    return pairwise_q_statistic(ctx["counts_stack"])

def _gp_q_statistic(ctx):
    return group_q_statistic(ctx["counts_stack"])

def _pw_output_correlation(ctx):
    return pairwise_output_correlation(ctx["probs_list"])

def _gp_output_correlation(ctx):
    return group_output_correlation(ctx["probs_list"])

def _pw_double_fault(ctx):
    return pairwise_double_fault(ctx["counts_stack"], ctx["N"])

def _gp_double_fault(ctx):
    return group_double_fault(ctx["counts_stack"], ctx["N"])

def _pw_jaccard(ctx):
    return pairwise_jaccard(ctx["counts_stack"])

def _gp_jaccard(ctx):
    return group_jaccard(ctx["counts_stack"])

def _pw_cohens_kappa(ctx):
    return pairwise_cohens_kappa(ctx["counts_stack"])

def _gp_cohens_kappa(ctx):
    return group_cohens_kappa(ctx["counts_stack"])

def _pw_correctness_disagreement(ctx):
    return pairwise_correctness_disagreement(ctx["counts_stack"])

def _gp_correctness_disagreement(ctx):
    return group_correctness_disagreement(ctx["counts_stack"])

def _pw_error_conditional_disagreement(ctx):
    return pairwise_error_conditional_disagreement(ctx["counts_stack"])

def _gp_error_conditional_disagreement(ctx):
    return group_error_conditional_disagreement(ctx["counts_stack"])

def _pw_overall_agreement(ctx):
    return pairwise_overall_agreement(ctx["counts_stack"])

def _gp_overall_agreement(ctx):
    return group_overall_agreement(ctx["counts_stack"])

def _pw_iou_top_n(ctx):
    return pairwise_iou_top_n(ctx["logits_list"])

def _gp_iou_top_n(ctx):
    return group_iou_top_n(ctx["logits_list"])


# Maps config metric name -> (pairwise_callable, group_callable)
# Pairwise callables return an (R, R) matrix; group callables return a scalar.
METRIC_REGISTRY: Dict[str, Tuple[Callable, Callable]] = {
    "pred_disagreement":              (_pw_disagreement, _gp_disagreement),
    "q_statistic":                    (_pw_q_statistic, _gp_q_statistic),
    "output_correlation":             (_pw_output_correlation, _gp_output_correlation),
    "double_fault":                   (_pw_double_fault, _gp_double_fault),
    "jaccard":                        (_pw_jaccard, _gp_jaccard),
    "cohens_kappa":                   (_pw_cohens_kappa, _gp_cohens_kappa),
    "correctness_disagreement":       (_pw_correctness_disagreement, _gp_correctness_disagreement),
    "error_conditional_disagreement": (_pw_error_conditional_disagreement, _gp_error_conditional_disagreement),
    "overall_agreement":              (_pw_overall_agreement, _gp_overall_agreement),
    "iou_top_n":                      (_pw_iou_top_n, _gp_iou_top_n),
}

# All ensemble methods supported by ensemble_probs_for_subset
ALL_ENSEMBLE_METHODS = ["soft", "hard", "max_confidence", "conf_weighted"]


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def get_ensemble_hash(run_names: List[str]) -> str:
    """Generate a consistent hash for a set of run names."""
    sorted_runs = sorted(run_names)
    return hashlib.sha256(json.dumps(sorted_runs).encode()).hexdigest()[:16]


def _update_index(save_dir: str, run_hash: str, run_names: List[str]) -> None:
    """Append an entry to the global index.json (create if needed)."""
    index_path = os.path.join(save_dir, "index.json")
    index = {}
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
# Subset evaluation
# ---------------------------------------------------------------------------

def evaluate_subset(
    subset_indices: Tuple[int, ...],
    subset_names: List[str],
    logits_matrix: torch.Tensor,
    labels: torch.Tensor,
    save_dir: str,
    metric_names: List[str],
    ensemble_methods: List[str],
) -> Dict[str, Any]:
    """
    Evaluate a specific subset of models.

    Args:
        subset_indices: Indices of models in the full pool.
        subset_names: Names of models in the subset.
        logits_matrix: Full logits matrix [R, N, C].
        labels: Ground truth labels [N].
        save_dir: Directory to save results.
        metric_names: Which diversity metrics to compute (keys into METRIC_REGISTRY).
        ensemble_methods: Which ensemble combination strategies to use.

    Returns:
        Dictionary of results (metrics + metadata).
    """
    # 1. Check cache
    run_hash = get_ensemble_hash(subset_names)
    cache_path = os.path.join(save_dir, f"ensemble_{run_hash}.pt")
    meta_path = os.path.join(save_dir, f"ensemble_{run_hash}.json")

    if os.path.exists(cache_path) and os.path.exists(meta_path):
        return torch.load(cache_path)

    # 2. Extract logits for this subset  [M, N, C]
    logits_subset = logits_matrix[list(subset_indices)]
    M = len(subset_indices)
    N = labels.numel()

    # 3. Compute Ensemble Accuracy
    ens_probs_map = ensemble_probs_for_subset(logits_subset)
    accuracies = {}
    for method in ensemble_methods:
        if method in ens_probs_map:
            preds = ens_probs_map[method].argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            accuracies[f"acc_{method}"] = acc

    # 4. Build metric context
    preds_list = [logits_subset[i].argmax(dim=1) for i in range(M)]
    probs_list = [torch.softmax(logits_subset[i], dim=1).cpu().numpy() for i in range(M)]
    logits_list = [logits_subset[i] for i in range(M)]
    counts_stack = pairwise_confusion_counts(preds_list, labels)

    ctx = {
        "preds_list": preds_list,
        "probs_list": probs_list,
        "logits_list": logits_list,
        "counts_stack": counts_stack,
        "labels": labels,
        "N": N,
    }

    # 5. Compute Diversity Metrics
    is_pairwise = (M == 2)
    div_metrics = {}
    for name in metric_names:
        if name not in METRIC_REGISTRY:
            continue
        pw_fn, gp_fn = METRIC_REGISTRY[name]
        if is_pairwise:
            mat = pw_fn(ctx)
            # For a pair, the scalar diversity is the off-diagonal value
            div_metrics[f"div_{name}"] = _average_off_diagonal(mat)
        else:
            div_metrics[f"div_{name}"] = gp_fn(ctx)

    # 6. Metadata
    metadata_list = [parse_model_name(name) for name in subset_names]

    results = {
        "acc": accuracies,
        "diversity": div_metrics,
        "run_names": subset_names,
        "hash": run_hash,
        "subset_size": M,
        "metadata": metadata_list,
    }

    # 7. Save
    torch.save(results, cache_path)
    with open(meta_path, "w") as f:
        json.dump({
            "hash": run_hash,
            "run_names": subset_names,
            "subset_size": M,
            "metrics_computed": list(div_metrics.keys()),
            "accuracy_methods": list(accuracies.keys()),
        }, f, indent=2)

    # 8. Update global index
    _update_index(save_dir, run_hash, subset_names)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Combinatorial Diversity Analysis")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit_subsets", type=int, default=0, help="Limit number of subsets per size k (0 = all)")
    args = parser.parse_args()

    # 1. Load Config
    config = load_config(args.config)

    model_dir = config.get("model_dir")
    if not model_dir:
        raise ValueError("Config must specify 'model_dir'")

    runs_to_load = config.get("runs")  # Optional list of specific runs
    metric_names = config.get("metrics", list(METRIC_REGISTRY.keys()))
    ensemble_methods = config.get("ensemble_methods", ALL_ENSEMBLE_METHODS)
    data_root = config.get("data_root", "./dataset")
    batch_size = config.get("batch_size", 100)
    save_dir = config.get("save_dir", "save/ensembles")
    min_k = config.get("min_k", 2)

    # Validate requested metrics
    unknown = [m for m in metric_names if m not in METRIC_REGISTRY]
    if unknown:
        print(f"Warning: unknown metrics ignored: {unknown}")
        metric_names = [m for m in metric_names if m in METRIC_REGISTRY]

    # 2. Load Models
    print(f"Loading models from: {model_dir}")
    bundles = []

    try:
        b = load_model_bundles(path=model_dir, device=args.device, eval_mode=True)
        bundles.extend(b)
    except FileNotFoundError:
        print(f"Direct load from {model_dir} failed. Checking subdirectories...")
        if os.path.exists(model_dir):
            subdirs = sorted([
                os.path.join(model_dir, d) for d in os.listdir(model_dir)
                if os.path.isdir(os.path.join(model_dir, d))
            ])
            for d in subdirs:
                try:
                    b = load_model_bundles(path=d, device=args.device, eval_mode=True)
                    bundles.extend(b)
                except FileNotFoundError:
                    continue

    # Filter to specific runs if requested
    if runs_to_load:
        filtered = []
        for b in bundles:
            rf = b.meta["run_folder"]
            rf_name = os.path.basename(rf)
            rf_parent = os.path.basename(os.path.dirname(rf))
            if any(target in (rf_name, rf_parent) for target in runs_to_load):
                filtered.append(b)
        bundles = filtered

    if not bundles:
        print("No models loaded.")
        return

    # Assign unique run names  (parent___trial)
    for b in bundles:
        parent = os.path.basename(os.path.dirname(b.meta["run_folder"]))
        trial = os.path.basename(b.meta["run_folder"])
        b.run_name = f"{parent}___{trial}"

    print(f"Loaded {len(bundles)} models.")

    # 3. Load Data (CIFAR-10 test set)
    images, labels = _load_clean_test_images(
        dataset_root=data_root,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=False,
    )
    images = _normalize_images(images)
    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=batch_size,
        shuffle=False,
    )

    # 4. Run inference (per-trial cached via run_inference utility)
    print("Running inference for all models (per-trial caching)...")
    pipeline_results = evaluate_bundles_individually(bundles, loader, force=False)

    if pipeline_results is None:
        print("No inference results obtained.")
        return

    # Align bundles with inference results
    cached_names = pipeline_results["run_names"]
    logits_matrix_all = pipeline_results["logits_matrix"]  # [R_total, N, C]

    indices_map = []
    valid_names = []
    for b in bundles:
        # run_inference uses _run_name(bundle.meta) which is Path(run_folder).name
        # Our b.run_name is parent___trial, but _run_name extracts basename of run_folder
        # So we need to match on what _run_name would return
        from pathlib import Path
        inferred_name = Path(b.meta["run_folder"]).name
        if inferred_name in cached_names:
            idx = cached_names.index(inferred_name)
            indices_map.append(idx)
            valid_names.append(b.run_name)
        elif b.run_name in cached_names:
            idx = cached_names.index(b.run_name)
            indices_map.append(idx)
            valid_names.append(b.run_name)
        else:
            print(f"Warning: Model {b.run_name} not found in inference results.")

    if not valid_names:
        print("No valid models found in inference results.")
        return

    logits_matrix = logits_matrix_all[indices_map]  # [R, N, C]
    labels = pipeline_results["labels_ref"]
    run_names = valid_names

    R = logits_matrix.shape[0]
    max_k = config.get("max_k", R)
    max_k = min(max_k, R)

    # 5. Combinatorial Loop
    os.makedirs(save_dir, exist_ok=True)
    all_results = []

    print(f"\nStarting combinatorial analysis: {R} models, k={min_k}..{max_k}")
    print(f"  Metrics: {metric_names}")
    print(f"  Ensemble methods: {ensemble_methods}")

    for k in range(min_k, max_k + 1):
        combos = list(combinations(range(R), k))

        if args.limit_subsets > 0 and len(combos) > args.limit_subsets:
            import random
            random.shuffle(combos)
            combos = combos[:args.limit_subsets]

        print(f"\nEvaluating {len(combos)} subsets of size k={k}")

        for combo_idx, indices in enumerate(combos):
            subset_names = [run_names[i] for i in indices]

            res = evaluate_subset(
                indices,
                subset_names,
                logits_matrix,
                labels,
                save_dir,
                metric_names,
                ensemble_methods,
            )
            all_results.append(res)

            if (combo_idx + 1) % 50 == 0:
                print(f"  [{combo_idx + 1}/{len(combos)}]")

    # 6. Correlation Analysis
    if not all_results:
        print("No results to analyze.")
        return

    # Collect all diversity and accuracy keys
    div_keys = sorted({k for r in all_results for k in r["diversity"].keys()})
    acc_keys = sorted({k for r in all_results for k in r["acc"].keys()})

    div_arrays = {k: [r["diversity"].get(k, float("nan")) for r in all_results] for k in div_keys}
    acc_arrays = {k: [r["acc"].get(k, float("nan")) for r in all_results] for k in acc_keys}

    import scipy.stats

    print(f"\n{'='*70}")
    print("Correlation Analysis (Pearson | Spearman)")
    print(f"{'='*70}")
    print(f"Total ensembles evaluated: {len(all_results)}")

    for acc_name in acc_keys:
        accs = acc_arrays[acc_name]
        print(f"\n  Target: {acc_name}")
        print(f"  {'Metric':<35} {'Pearson':>8} {'Spearman':>9}  {'N':>5}")
        print(f"  {'-'*60}")

        for div_name in div_keys:
            divs = div_arrays[div_name]
            # Filter NaN
            valid = [(a, d) for a, d in zip(accs, divs) if np.isfinite(a) and np.isfinite(d)]
            if len(valid) < 3:
                print(f"  {div_name:<35} {'N/A':>8} {'N/A':>9}  {len(valid):>5}")
                continue
            va, vd = zip(*valid)
            pearson = scipy.stats.pearsonr(va, vd)[0]
            spearman = scipy.stats.spearmanr(va, vd)[0]
            print(f"  {div_name:<35} {pearson:>+8.4f} {spearman:>+9.4f}  {len(valid):>5}")

    print(f"\n{'='*70}")
    print(f"Results saved to: {save_dir}/")
    print(f"Index file: {os.path.join(save_dir, 'index.json')}")


if __name__ == "__main__":
    main()
