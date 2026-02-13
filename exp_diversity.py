"""
Ensemble Diversity Analysis — unified diversity and combinatorial evaluation.

This script is the single entry-point for **all** ensemble diversity workflows.
It supersedes the old ``exp_combinatorial.py`` (now deleted).

Three modes
-----------

``--model <path>``
    Load all trials of a single model directory, compute pairwise diversity
    among them and ensemble accuracy.  Equivalent to the old ``exp_diversity.py --path``.

``--config <yaml>``
    Load models described in a YAML config (same format as the old
    ``exp_combinatorial.py``).  Group trials by parent model and compute
    **within-model** pairwise diversity + ensemble accuracy for each group.

``--config <yaml> --combinatorial``
    Combinatorial mode.  Iterate all subsets of size ``min_k`` …  ``max_k``
    across the full pool of models and evaluate each ensemble.  Produces
    the same per-subset ``.pt`` files as the old ``exp_combinatorial.py``
    plus a correlation analysis at the end.

CLI Synopsis
~~~~~~~~~~~~
::

    # Single model dir  (all trials)
    python exp_diversity.py --model save/ResNet18/models/<model_dir> \\
        --use_cache --print_all

    # Config-driven, within-model diversity
    python exp_diversity.py --config configs/test_combinatorial.yaml --use_cache

    # Config-driven, combinatorial across models
    python exp_diversity.py --config configs/test_combinatorial.yaml \\
        --combinatorial --limit_subsets 5 --use_cache

YAML Config Fields
------------------
::

    model_dir: "save/ResNet18/models"       # root containing model folders
    runs:                                    # (optional) restrict to these folders
      - "crossentropy_wstopo_torus_..."
    data_root: "dataset"                     # CIFAR-10 location
    batch_size: 100
    save_dir: "save/ensembles"               # where ensemble results are saved
    min_k: 2                                 # smallest subset size  (combinatorial)
    max_k: 3                                 # largest subset size   (combinatorial)
    metrics:                                 # (optional) diversity metrics to compute
      - pred_disagreement
      - q_statistic
    ensemble_methods:                        # (optional) ensemble strategies
      - soft
      - hard

Output Files
============

Filesystem layout (cache)
-------------------------
All modes write to a single flat directory (``save_dir``, default ``save/ensembles``).
Each evaluated ensemble produces **three** artefacts::

    <save_dir>/
      ensemble_<hash>.pt     ← torch dict  (main result)
      ensemble_<hash>.json   ← sidecar JSON (lightweight metadata)
      index.json             ← global lookup  hash → run_names

The directory acts as a **content-addressed cache**: before computing any
ensemble, the script checks whether ``ensemble_<hash>.pt`` already exists
and skips recomputation when found.  Results produced by ``--model``,
``--config``, or ``--config --combinatorial`` are **identical** for the same
set of component run names, so work done in one mode is automatically reused
by the others.

Hash → ensemble components
~~~~~~~~~~~~~~~~~~~~~~~~~~
The hash is a deterministic **SHA-256** (truncated to 16 hex chars) of the
**sorted** JSON-encoded list of run name strings that form the ensemble::

    import hashlib, json
    run_hash = hashlib.sha256(json.dumps(sorted(run_names)).encode()).hexdigest()[:16]

You can reverse-lookup components in three ways::

    # 1. Global index  (all ensembles)
    with open("save/ensembles/index.json") as f:
        index = json.load(f)
    components = index["273be8d129eb1dce"]

    # 2. Per-ensemble sidecar JSON
    with open("save/ensembles/ensemble_273be8d129eb1dce.json") as f:
        meta = json.load(f)
    components = meta["run_names"]

    # 3. From the .pt file
    data = torch.load("save/ensembles/ensemble_273be8d129eb1dce.pt")
    components = data["run_names"]

Internal ``.pt`` structure
--------------------------
Every ``.pt`` file contains a single dictionary with the **same** schema,
regardless of which CLI flag produced it::

    {
        "acc": {                                 # ensemble accuracies
            "acc_soft":           float,
            "acc_hard":           float,
            "acc_max_confidence": float,
            "acc_conf_weighted":  float,
        },
        "diversity": {                           # scalar summary per metric
            "div_pred_disagreement": float,      #   off-diagonal mean of pairwise matrix
            "div_q_statistic":       float,
            ...
        },
        "pairwise": {                            # full R×R pairwise matrices
            "pw_pred_disagreement": np.ndarray,  #   shape (M, M)
            "pw_q_statistic":       np.ndarray,
            ...
        },
        "run_names":              List[str],     # sorted component run names
        "hash":                   str,           # 16-hex SHA-256
        "subset_size":            int,           # M (number of models)
        "metadata":               List[dict],    # parsed hyperparams per model
        "individual_accuracies":  List[float],   # per-model test accuracy
    }

Sidecar ``.json`` structure
---------------------------
::

    {
        "hash":              str,
        "run_names":         List[str],
        "subset_size":       int,
        "metrics_computed":  List[str],   # keys in ``diversity``
        "accuracy_methods":  List[str],   # keys in ``acc``
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Repo-local imports
from utils.experiments import CIFAR10_MEAN, CIFAR10_STD, load_config, parse_model_name
from utils.load import load_model_bundles
from utils.run_inference import evaluate_bundles_individually
from utils.ensemble_combination import ensemble_probs_for_subset, get_ensemble_hash, update_index
from utils.metric_registry import METRIC_REGISTRY, ALL_ENSEMBLE_METHODS
from utils.metrics import pairwise_confusion_counts, _average_off_diagonal
from exp_errorcorr import _load_clean_test_images, _normalize_images, _run_name


# ---------------------------------------------------------------------------
# Ensemble evaluation  (shared by all modes)
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
    Evaluate a specific subset (ensemble) of models.

    1. Check the on-disk cache (``ensemble_<hash>.pt``).  Return early on hit.
    2. Slice the logits matrix to this subset.
    3. Compute ensemble accuracy for each requested method.
    4. Compute pairwise diversity matrices and their scalar (off-diagonal mean)
       summaries for each requested metric.
    5. Compute individual per-model accuracy.
    6. Save results to ``.pt`` + sidecar ``.json`` and update ``index.json``.

    Args:
        subset_indices: Indices into *logits_matrix* (dim 0) for this ensemble.
        subset_names:   Unique run names (used for hashing and in the saved file).
        logits_matrix:  Full pool logits ``[R, N, C]``.
        labels:         Ground-truth labels ``[N]``.
        save_dir:       Directory to save/load cached results.
        metric_names:   Keys into ``METRIC_REGISTRY`` — which metrics to compute.
        ensemble_methods: Ensemble combination strategies (e.g. ``["soft", "hard"]``).

    Returns:
        The result dictionary (same schema as the ``.pt`` file).
    """
    # ------------------------------------------------------------------
    # 1. Cache check
    # ------------------------------------------------------------------
    run_hash = get_ensemble_hash(subset_names)
    os.makedirs(save_dir, exist_ok=True)
    cache_path = os.path.join(save_dir, f"ensemble_{run_hash}.pt")
    meta_path = os.path.join(save_dir, f"ensemble_{run_hash}.json")

    if os.path.exists(cache_path) and os.path.exists(meta_path):
        return torch.load(cache_path, weights_only=False)

    # ------------------------------------------------------------------
    # 2. Slice logits
    # ------------------------------------------------------------------
    logits_subset = logits_matrix[list(subset_indices)]  # [M, N, C]
    M = len(subset_indices)
    N = labels.numel()

    # ------------------------------------------------------------------
    # 3. Ensemble accuracy
    # ------------------------------------------------------------------
    ens_probs = ensemble_probs_for_subset(logits_subset)
    accuracies: Dict[str, float] = {}
    for method in ensemble_methods:
        if method in ens_probs:
            preds = ens_probs[method].argmax(dim=1)
            accuracies[f"acc_{method}"] = float((preds == labels).float().mean().item())

    # ------------------------------------------------------------------
    # 4. Diversity metrics
    # ------------------------------------------------------------------
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

    div_scalars: Dict[str, float] = {}
    pw_matrices: Dict[str, np.ndarray] = {}

    for name in metric_names:
        if name not in METRIC_REGISTRY:
            continue
        pw_fn, gp_fn = METRIC_REGISTRY[name]
        mat = pw_fn(ctx)
        pw_matrices[f"pw_{name}"] = mat
        div_scalars[f"div_{name}"] = _average_off_diagonal(mat)

    # ------------------------------------------------------------------
    # 5. Individual accuracies
    # ------------------------------------------------------------------
    indiv_accs = [
        float((preds_list[i] == labels).float().mean().item()) for i in range(M)
    ]

    # ------------------------------------------------------------------
    # 6. Metadata & save
    # ------------------------------------------------------------------
    metadata_list = [parse_model_name(n) for n in subset_names]

    results: Dict[str, Any] = {
        "acc": accuracies,
        "diversity": div_scalars,
        "pairwise": pw_matrices,
        "run_names": sorted(subset_names),
        "hash": run_hash,
        "subset_size": M,
        "metadata": metadata_list,
        "individual_accuracies": indiv_accs,
    }

    torch.save(results, cache_path)

    with open(meta_path, "w") as f:
        json.dump(
            {
                "hash": run_hash,
                "run_names": sorted(subset_names),
                "subset_size": M,
                "metrics_computed": list(div_scalars.keys()),
                "accuracy_methods": list(accuracies.keys()),
            },
            f,
            indent=2,
        )

    update_index(save_dir, run_hash, subset_names)
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_run_name(bundle) -> str:
    """
    Build a unique run name from a bundle's metadata.

    Format: ``<parent_model_dir>___<trial_folder>``
    e.g. ``crossentropy_wstopo_torus_256embdims_0.008rho_...___trial_00``
    """
    parent = os.path.basename(os.path.dirname(bundle.meta["run_folder"]))
    trial = os.path.basename(bundle.meta["run_folder"])
    return f"{parent}___{trial}"


def _map_bundles_to_inference(
    bundles: list,
    pipeline_results: Dict[str, Any],
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """
    Align loaded bundles with inference results (which may use different naming).

    Returns:
        (run_names, logits_matrix, labels)
        where run_names are full ``parent___trial`` names and the logits/labels
        tensors are re-indexed to match.
    """
    cached_names: List[str] = pipeline_results["run_names"]

    indices_map: List[int] = []
    valid_names: List[str] = []

    for b in bundles:
        full_name = _build_run_name(b)
        inferred_name = Path(b.meta["run_folder"]).name

        if inferred_name in cached_names:
            indices_map.append(cached_names.index(inferred_name))
            valid_names.append(full_name)
        elif full_name in cached_names:
            indices_map.append(cached_names.index(full_name))
            valid_names.append(full_name)
        else:
            print(f"Warning: Model {full_name} not found in inference results, skipping.")

    if not valid_names:
        raise RuntimeError("No valid models found in inference results.")

    logits_matrix = pipeline_results["logits_matrix"][indices_map]
    labels = pipeline_results["labels_ref"]
    return valid_names, logits_matrix, labels


def _load_data(
    data_root: str = "./dataset",
    batch_size: int = 100,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> Tuple[DataLoader, torch.Tensor]:
    """Load CIFAR-10 test images, normalise, and return (DataLoader, raw_labels)."""
    images, labels = _load_clean_test_images(
        dataset_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    images = _normalize_images(images)
    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader, labels


# ---------------------------------------------------------------------------
# Printing utilities
# ---------------------------------------------------------------------------

def _fmt_mat(mat: np.ndarray, names: List[str]) -> str:
    """Format an R×R matrix as an aligned text block."""
    R = len(names)
    lines = []
    for i in range(R):
        row = " ".join(
            f"{mat[i, j]:.4f}" if np.isfinite(mat[i, j]) else "  nan "
            for j in range(R)
        )
        lines.append(row)
    return "\n".join(lines)


def _print_subset_result(
    result: Dict[str, Any],
    print_all: bool = False,
) -> None:
    """Pretty-print the evaluation result for one ensemble."""
    names = result["run_names"]
    M = result["subset_size"]
    indiv = result.get("individual_accuracies", [])

    print(f"\n{'─' * 60}")
    print(f"Ensemble  ({M} models)   hash={result['hash']}")
    for i, n in enumerate(names):
        acc_str = f"  acc={indiv[i]:.4f}" if i < len(indiv) else ""
        print(f"  {n}{acc_str}")

    print("\nEnsemble accuracies:")
    for k, v in result["acc"].items():
        print(f"  {k}: {v:.4f}")

    print("\nDiversity (scalar summaries):")
    for k, v in result["diversity"].items():
        print(f"  {k}: {v:.4f}" if np.isfinite(v) else f"  {k}: nan")

    if print_all and result.get("pairwise"):
        print("\nPairwise matrices:")
        for k, mat in result["pairwise"].items():
            print(f"\n  {k}:")
            print("  " + _fmt_mat(mat, names).replace("\n", "\n  "))




# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def mode_model(args) -> None:
    """``--model`` mode: pairwise diversity among all trials of one model dir."""
    bundles = load_model_bundles(
        path=args.model,
        device=args.device,
        eval_mode=True,
    )
    if not bundles:
        print("No trials found.")
        return

    for b in bundles:
        b.run_name = _build_run_name(b)

    loader, _ = _load_data(
        data_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    pipeline = evaluate_bundles_individually(bundles, loader, force=not args.use_cache)
    if pipeline is None:
        print("No inference results obtained.")
        return

    run_names, logits_matrix, labels = _map_bundles_to_inference(bundles, pipeline)

    result = evaluate_subset(
        subset_indices=tuple(range(len(run_names))),
        subset_names=run_names,
        logits_matrix=logits_matrix,
        labels=labels,
        save_dir=args.save_dir,
        metric_names=args.metrics,
        ensemble_methods=args.ensemble_methods,
    )

    _print_subset_result(result, print_all=args.print_all)
    print(f"\nResult saved to: {args.save_dir}/ensemble_{result['hash']}.pt")


def mode_config(args) -> None:
    """``--config`` (no ``--combinatorial``): within-model diversity per group."""
    config = load_config(args.config)

    model_dir = config.get("model_dir")
    if not model_dir:
        raise ValueError("Config must specify 'model_dir'")

    runs_filter = config.get("runs")
    metric_names = args.metrics  # CLI override or config
    ensemble_methods = args.ensemble_methods
    data_root = config.get("data_root", args.dataset_root)
    batch_size = config.get("batch_size", args.batch_size)
    save_dir = args.save_dir

    # --- load bundles ---
    bundles = _load_bundles_from_config(model_dir, runs_filter, args.device)
    if not bundles:
        print("No models loaded.")
        return

    for b in bundles:
        b.run_name = _build_run_name(b)

    # --- data ---
    loader, _ = _load_data(data_root=data_root, batch_size=batch_size)

    # --- inference ---
    print("Running inference for all models (per-trial caching)...")
    pipeline = evaluate_bundles_individually(bundles, loader, force=not args.use_cache)
    if pipeline is None:
        print("No inference results obtained.")
        return

    run_names, logits_matrix, labels = _map_bundles_to_inference(bundles, pipeline)

    # --- group by parent model ---
    # Build a mapping: parent → list of (index_in_run_names, full_name)
    parent_groups: Dict[str, List[Tuple[int, str]]] = {}
    for idx, name in enumerate(run_names):
        parent = name.split("___")[0] if "___" in name else name
        parent_groups.setdefault(parent, []).append((idx, name))

    all_results = []
    for parent, members in parent_groups.items():
        indices = tuple(m[0] for m in members)
        names = [m[1] for m in members]
        print(f"\n{'=' * 60}")
        print(f"Model group: {parent}  ({len(names)} trials)")

        result = evaluate_subset(
            subset_indices=indices,
            subset_names=names,
            logits_matrix=logits_matrix,
            labels=labels,
            save_dir=save_dir,
            metric_names=metric_names,
            ensemble_methods=ensemble_methods,
        )
        all_results.append(result)
        _print_subset_result(result, print_all=args.print_all)

    print(f"\nResults saved to: {save_dir}/")


def mode_combinatorial(args) -> None:
    """``--config --combinatorial``: exhaustive subset enumeration."""
    config = load_config(args.config)

    model_dir = config.get("model_dir")
    if not model_dir:
        raise ValueError("Config must specify 'model_dir'")

    runs_filter = config.get("runs")
    metric_names = args.metrics
    ensemble_methods = args.ensemble_methods
    data_root = config.get("data_root", args.dataset_root)
    batch_size = config.get("batch_size", args.batch_size)
    save_dir = args.save_dir
    min_k = config.get("min_k", 2)

    # --- load bundles ---
    bundles = _load_bundles_from_config(model_dir, runs_filter, args.device)
    if not bundles:
        print("No models loaded.")
        return

    for b in bundles:
        b.run_name = _build_run_name(b)

    print(f"Loaded {len(bundles)} models.")

    # --- data ---
    loader, _ = _load_data(data_root=data_root, batch_size=batch_size)

    # --- inference ---
    print("Running inference for all models (per-trial caching)...")
    pipeline = evaluate_bundles_individually(bundles, loader, force=not args.use_cache)
    if pipeline is None:
        print("No inference results obtained.")
        return

    run_names, logits_matrix, labels = _map_bundles_to_inference(bundles, pipeline)
    R = len(run_names)
    max_k = config.get("max_k", R)
    max_k = min(max_k, R)

    # --- combinatorial loop ---
    os.makedirs(save_dir, exist_ok=True)
    all_results: List[Dict[str, Any]] = []

    print(f"\nCombinatorial analysis: {R} models, k={min_k}..{max_k}")
    print(f"  Metrics: {metric_names}")
    print(f"  Ensemble methods: {ensemble_methods}")

    for k in range(min_k, max_k + 1):
        combos = list(combinations(range(R), k))

        if args.limit_subsets > 0 and len(combos) > args.limit_subsets:
            import random
            random.shuffle(combos)
            combos = combos[: args.limit_subsets]

        print(f"\nEvaluating {len(combos)} subsets of size k={k}")

        for combo_idx, indices in enumerate(combos):
            subset_names = [run_names[i] for i in indices]

            res = evaluate_subset(
                subset_indices=indices,
                subset_names=subset_names,
                logits_matrix=logits_matrix,
                labels=labels,
                save_dir=save_dir,
                metric_names=metric_names,
                ensemble_methods=ensemble_methods,
            )
            all_results.append(res)

            if (combo_idx + 1) % 50 == 0:
                print(f"  [{combo_idx + 1}/{len(combos)}]")

    # --- correlation analysis ---
    if all_results:
        _run_correlation_analysis(all_results)
    else:
        print("No results to analyse.")

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {save_dir}/")
    print(f"Index file: {os.path.join(save_dir, 'index.json')}")


# ---------------------------------------------------------------------------
# Bundle loading helper (for --config modes)
# ---------------------------------------------------------------------------

def _load_bundles_from_config(
    model_dir: str,
    runs_filter: Optional[List[str]],
    device: str,
) -> list:
    """
    Load model bundles from *model_dir*, optionally filtered to *runs_filter*.

    Tries loading from *model_dir* directly first; if that fails, iterates
    its subdirectories (each subdir is expected to be a model folder containing
    trial sub-folders).
    """
    bundles: list = []

    try:
        bundles.extend(load_model_bundles(path=model_dir, device=device, eval_mode=True))
    except FileNotFoundError:
        print(f"Direct load from {model_dir} failed. Scanning subdirectories...")
        if os.path.exists(model_dir):
            for d in sorted(os.listdir(model_dir)):
                full = os.path.join(model_dir, d)
                if os.path.isdir(full):
                    try:
                        bundles.extend(load_model_bundles(path=full, device=device, eval_mode=True))
                    except FileNotFoundError:
                        continue

    if runs_filter:
        filtered = []
        for b in bundles:
            rf = b.meta["run_folder"]
            rf_name = os.path.basename(rf)
            rf_parent = os.path.basename(os.path.dirname(rf))
            if any(target in (rf_name, rf_parent) for target in runs_filter):
                filtered.append(b)
        bundles = filtered

    return bundles


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ensemble Diversity Analysis (pairwise & combinatorial).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Mutually-exclusive input source ---
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a model directory containing trial sub-folders.",
    )
    source.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file (same format as old exp_combinatorial).",
    )

    # --- Mode modifier ---
    parser.add_argument(
        "--combinatorial",
        action="store_true",
        default=False,
        help=(
            "With --config, enumerate all subsets of size min_k..max_k "
            "and evaluate each. Requires --config."
        ),
    )
    parser.add_argument(
        "--limit_subsets",
        type=int,
        default=0,
        help="Limit the number of subsets per k in combinatorial mode (0 = all).",
    )

    # --- Metric / ensemble overrides ---
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help=(
            "Override which diversity metrics to compute. "
            "Defaults to all registered metrics (or those listed in the config)."
        ),
    )
    parser.add_argument(
        "--ensemble_methods",
        nargs="*",
        default=None,
        help="Override ensemble combination methods. Default: all.",
    )

    # --- Common options ---
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_cache", action="store_true", default=False, help="Use cached inference results.")
    parser.add_argument("--print_all", action="store_true", default=False, help="Print full pairwise matrices.")
    parser.add_argument("--save_dir", type=str, default=None, help="Override save directory (default: save/ensembles).")
    parser.add_argument("--dataset_root", type=str, default="./dataset", help="CIFAR-10 dataset root.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    # --- Validation ---
    if args.combinatorial and args.config is None:
        parser.error("--combinatorial requires --config.")

    # --- Resolve metrics & ensemble methods from config + CLI ---
    config: Dict[str, Any] = {}
    if args.config is not None:
        config = load_config(args.config)

    if args.metrics is None:
        args.metrics = config.get("metrics", list(METRIC_REGISTRY.keys()))
    if args.ensemble_methods is None:
        args.ensemble_methods = config.get("ensemble_methods", list(ALL_ENSEMBLE_METHODS))
    if args.save_dir is None:
        args.save_dir = config.get("save_dir", "save/ensembles")

    # Warn about unknown metrics
    unknown = [m for m in args.metrics if m not in METRIC_REGISTRY]
    if unknown:
        print(f"Warning: unknown metrics ignored: {unknown}")
        args.metrics = [m for m in args.metrics if m in METRIC_REGISTRY]

    # --- Dispatch ---
    if args.model is not None:
        mode_model(args)
    elif args.combinatorial:
        mode_combinatorial(args)
    else:
        mode_config(args)


if __name__ == "__main__":
    main()
