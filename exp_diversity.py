"""
Evaluate performance agreement between model trials.

This script re-uses the model-loading helpers from ``exp_errorcorr.py``.
For each model trial it collects per-sample predictions and logits
(via the same data-loading pipeline), then computes pairwise metrics
using individual functions from ``utils.metrics``.

Prediction-based metrics (from argmax labels):
  - asym_ratio, correctness_disagreement, error_conditional_disagreement
  - overall agreement rate, Jaccard index, Cohen's kappa
  - McNemar's exact two-sided p-value

Logit / probability-based metrics:
  - pred_disagreement, norm_pred_disagreement (per ensemble method)
  - double_fault, output_correlation, Q statistic

Parameter-based metrics (optional, --analysis params or both):
  - param_cosine: cosine similarity between flattened parameter vectors.

Usage:
    python exp_performance.py --path <model_dir> [--use_cache] [--analysis logits|params|both] [--print_all]

Notes:
  - Imports helpers from ``exp_errorcorr.py`` (data loading, evaluation).
  - All pairwise metric functions live in ``utils/metrics.py``.
  - Evaluates on the clean CIFAR-10 test set (no noise).
"""

from __future__ import annotations

from itertools import combinations
import os
import sys
from typing import Tuple, Dict, List, Optional, Union

import numpy as np
import torch

from utils.metrics import (
    pairwise_confusion_counts,
    pairwise_asym_ratio,
    pairwise_asym_ratio_reverse,
    pairwise_correctness_disagreement,
    pairwise_error_conditional_disagreement,
    pairwise_overall_agreement,
    pairwise_cohens_kappa,
    pairwise_jaccard,
    pairwise_mcnemar_p,
    pairwise_pred_disagreement,
    pairwise_norm_pred_disagreement,
    pairwise_double_fault,
    pairwise_output_correlation,
    pairwise_q_statistic,
    pairwise_param_cosine,
)

from utils.ensemble_combination import compute_all_ensemble_probs
from utils.run_inference import evaluate_bundles_individually

# Reuse existing helpers from the repository.
# These are local modules in the repo; importing is fine when adding the file to the repo root.
try:
    from utils.load import parse_model_load_args, load_model_bundles
    from utils.experiments import CIFAR10_MEAN, CIFAR10_STD

    # Helpers that live in exp_errorcorr.py
    from exp_errorcorr import (
        _collect_errors_and_preds,
        _load_clean_test_images,
        _run_name,
    )
except Exception as e:
    raise ImportError(
        "Failed to import repository helpers. Ensure you're running this from the repo root "
        "and that `exp_errorcorr.py` is present. Original error: " + str(e)
    )


def _print_pairwise_summary(
    metrics: Dict[str, np.ndarray], run_names: List[str]
) -> None:
    """
    Print compact pairwise summaries for the most relevant metrics.
    Shows small matrices and per-pair diagnostics.
    """
    R = len(run_names)

    def fmt_mat(mat: np.ndarray) -> str:
        lines = []
        for i in range(R):
            row = " ".join(
                f"{mat[i,j]:.4f}" if np.isfinite(mat[i, j]) else " nan "
                for j in range(R)
            )
            lines.append(row)
        return "\n".join(lines)

    print("\nPairwise asymmetric ratio (A_correct_B_incorrect / N):")
    print(fmt_mat(metrics["asym_ab"]))

    print("\nPairwise correctness disagreement (one correct, one wrong / N):")
    print(fmt_mat(metrics["correctness_disagreement"]))

    print("\nPairwise overall agreement rate (both correct or both incorrect):")
    print(fmt_mat(metrics["overall_agree"]))

    print("\nPairwise Cohen's kappa:")
    print(fmt_mat(metrics["cohens_kappa"]))

    print("\nPairwise Jaccard index over correct-sets:")
    print(fmt_mat(metrics["jaccard"]))

    print("\nPairwise error-conditional disagreement (N01+N10)/(N01+N10+N00):")
    print(fmt_mat(metrics["error_conditional_disagreement"]))

    print("\nPairwise McNemar two-sided p-values (b vs c):")
    print(fmt_mat(metrics["mcnemar_p"]))





def _print_ensemble_summary(
    ensemble_probs: Dict[Union[str, Tuple[int, int]], Dict[str, torch.Tensor]],
    labels: torch.Tensor,
    run_names: List[str],
) -> None:
    """Print a summary table of ensemble accuracies derived from cached prob tensors."""
    methods = ["soft", "hard", "max_confidence", "conf_weighted"]

    def _acc(probs_tensor: torch.Tensor) -> float:
        preds = probs_tensor.argmax(dim=1)
        return float((preds == labels).float().mean().item())

    # Full ensemble
    print("\n  Ensemble accuracies (all models):")
    all_probs = ensemble_probs["all"]
    for m in methods:
        print(f"    {m}: {_acc(all_probs[m]):.4f}")

    # Pairwise ensembles
    print("\n  Pairwise ensemble accuracies:")
    for key in sorted(k for k in ensemble_probs if k != "all"):
        i, j = key
        name_i = run_names[i] if i < len(run_names) else str(i)
        name_j = run_names[j] if j < len(run_names) else str(j)
        pair_probs = ensemble_probs[key]
        accs = " | ".join(f"{m}: {_acc(pair_probs[m]):.4f}" for m in methods)
        print(f"    ({name_i}, {name_j})  {accs}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate performance agreement between model trials."
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        default=False,
        help="Use cache for evaluation results if available.",
    )
    parser.add_argument(
        "--analysis",
        choices=["logits", "params", "both"],
        default="logits",
        help="Type of analysis to perform: logits (default), params, or both",
    )
    parser.add_argument(
        "--print_all",
        action="store_true",
        default=False,
        help="Print all pairwise matrices, not just summary",
    )
    args_partial, remaining = parser.parse_known_args()
    # Feed only the remaining args to parse_model_load_args so it doesn't
    # choke on --use_cache / --analysis / --print_all.
    orig_argv = sys.argv
    sys.argv = [sys.argv[0]] + remaining
    args = parse_model_load_args()
    sys.argv = orig_argv
    args.use_cache = args_partial.use_cache
    args.analysis = args_partial.analysis
    args.print_all = args_partial.print_all
    bundles = load_model_bundles(
        path=args.path,
        prefer=args.prefer,
        device=args.device,
        dp_if_multi_gpu=args.dp,
        eval_mode=True,
        strict=True,
    )

    pin_memory = torch.cuda.is_available()
    base_images, base_labels = _load_clean_test_images(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # Build a clean data loader (no noise)
    from torch.utils.data import TensorDataset, DataLoader
    from exp_errorcorr import _normalize_images

    normalized_images = _normalize_images(base_images)
    loader = DataLoader(
        TensorDataset(normalized_images, base_labels),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    result = evaluate_bundles_individually(bundles, loader, force=not args.use_cache)
    if result is None:
        print("No trials evaluated.")
        return

    run_names = result["run_names"]
    accuracies = result["accuracies"]
    num_trials = len(run_names)

    # Determine which analyses to run
    do_logits = args.analysis in ("logits", "both")
    do_params = args.analysis in ("params", "both")

    metrics_logits = None

    if do_logits:
        labels = result["labels_ref"]
        logits_matrix = result.get("logits_matrix")

        # Extract predictions and probabilities
        R = logits_matrix.shape[0]
        N = labels.numel()
        preds_list = [logits_matrix[i].argmax(dim=1).cpu() for i in range(R)]
        probs_list = [
            torch.softmax(logits_matrix[i], dim=1).cpu().numpy() for i in range(R)
        ]

        # Compute ensemble probs first (needed by norm_pred_disagreement)
        cache_dir = args.path
        ensemble_cache_dir = cache_dir if args.use_cache else None
        ensemble_probs = compute_all_ensemble_probs(
            logits_matrix=logits_matrix,
            labels=labels,
            run_names=run_names,  # New argument
            cache_dir=ensemble_cache_dir,
        )

        # Compute individual metrics
        counts_stack = pairwise_confusion_counts(preds_list, labels)
        pred_dis = pairwise_pred_disagreement(preds_list)

        metrics_logits = {
            "asym_ab": pairwise_asym_ratio(counts_stack),
            "asym_ba": pairwise_asym_ratio_reverse(counts_stack),
            "correctness_disagreement": pairwise_correctness_disagreement(counts_stack),
            "error_conditional_disagreement": pairwise_error_conditional_disagreement(
                counts_stack
            ),
            "overall_agree": pairwise_overall_agreement(counts_stack),
            "cohens_kappa": pairwise_cohens_kappa(counts_stack),
            "jaccard": pairwise_jaccard(counts_stack),
            "mcnemar_p": pairwise_mcnemar_p(counts_stack),
            "pred_disagreement": pred_dis,
            "norm_pred_disagreement": pairwise_norm_pred_disagreement(
                pred_dis, ensemble_probs, labels
            ),
            "double_fault": pairwise_double_fault(counts_stack, N),
            "output_correlation": pairwise_output_correlation(probs_list),
            "q_statistic": pairwise_q_statistic(counts_stack),
        }

    if do_params:
        param_vecs = result.get("param_vecs")
        if param_vecs is None:
            print("No parameter vectors found.")
        else:
            metrics_params = {"param_cosine": pairwise_param_cosine(param_vecs)}

    # Print diagnostics
    print(f"\nDiagnostics (clean, {num_trials} trials):")
    print("Individual accuracies:")
    for rn, a in zip(run_names, accuracies):
        print(f"  {rn}: {a:.4f}")

    def fmt_mat(mat: np.ndarray) -> str:
        R = len(run_names)
        lines = []
        for i in range(R):
            row = " ".join(
                f"{mat[i,j]:.4f}" if np.isfinite(mat[i, j]) else " nan "
                for j in range(R)
            )
            lines.append(row)
        return "\n".join(lines)

    if do_logits and metrics_logits is not None:
        print("\n[Logits-based metrics]")
        _print_pairwise_summary(metrics_logits, run_names)

        print("\n[Ensemble probability summary]")
        _print_ensemble_summary(ensemble_probs, labels, run_names)

        if args.print_all:
            print(
                "\nPairwise Prediction Disagreement (fraction predicted labels differ):"
            )
            print(fmt_mat(metrics_logits["pred_disagreement"]))

            # norm_pred_disagreement is a dict of matrices, one per ensemble method
            norm_dis = metrics_logits["norm_pred_disagreement"]
            for method, mat in norm_dis.items():
                print(f"\nNormalised Pred Disagreement (ensemble={method}):")
                print(fmt_mat(mat))

            print("\nPairwise Double-Fault (both incorrect fraction):")
            print(fmt_mat(metrics_logits["double_fault"]))

            print("\nPairwise Output Correlation:")
            print(fmt_mat(metrics_logits["output_correlation"]))

            print("\nPairwise Q Statistic:")
            print(fmt_mat(metrics_logits["q_statistic"]))

    if do_params and metrics_params is not None:
        print("\n[Parameter-based metrics]")
        print("\nPairwise Parameter Cosine Similarity:")
        print(fmt_mat(metrics_params["param_cosine"]))


if __name__ == "__main__":
    main()
