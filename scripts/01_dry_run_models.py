#!/usr/bin/env python3
"""
01_dry_run_models.py — Training Idempotency Dry Run

Performs the same config resolution as 01_train_models.py (same Hydra
decorator, same validate_training_config / resolve_seed / cfg_hash /
find_finished_model_run) so the computed hash is guaranteed identical to
what the real training script would produce.

If no matching FINISHED run exists, fetches candidate runs from MLflow and
prints a param diff so you can immediately see what differs — and which
differing params are the ones that drive the identity hash (marked *).

Use dry_filter.* overrides to narrow the candidate search:

    uv run python scripts/01_dry_run_models.py \\
        loss.rho=0.008 loss.topology=torus trial=0 \\
        dry_filter.topology=torus dry_filter.trial=0
"""

from __future__ import annotations

from datetime import datetime

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.config.hash import cfg_hash, model_identity_fields
from src.config.validation import validate_training_config
from src.mlflow_utils import resolve_seed, setup_mlflow
from src.repositories.functional_run_repository import (
    configure_run_repository,
    find_finished_model_run,
    search_runs,
)


# ── Identity-field detection ──────────────────────────────────────────────────

def _is_identity_param(mlflow_key: str, identity_fields: set[str]) -> bool:
    """Return True if this MLflow param name traces back to an identity field.

    MLflow stores params with their leaf name (e.g. 'rho', not 'loss.rho').
    identity_fields contains dot-paths like 'loss.rho', 'model.arch', 'seed'.
    We match if any identity field's leaf equals the mlflow key.
    """
    for field in identity_fields:
        leaf = field.rsplit(".", 1)[-1]
        if leaf == mlflow_key:
            return True
    return False


# ── Proposed param dict (mirrors 01_train_models.py lines 289–336) ───────────

def _build_proposed_params(cfg: DictConfig, seed: int) -> dict[str, str]:
    """Build the same param dict that schema_log_params would log.

    Mirrors 01_train_models.py lines 289–336 exactly.
    None values are filtered out (matching _clean_params in the schema logger).
    """
    raw = {
        "rho": float(cfg.loss.rho),
        "seed": seed,
        "epochs": cfg.training.epochs,
        "batch_size": cfg.training.batch_size,
        "learning_rate": cfg.training.learning_rate,
        "optimiser": cfg.training.optimiser,
        "weight_decay": cfg.training.weight_decay,
        "momentum": cfg.training.momentum,
        "scheduler": cfg.training.scheduler,
        "amp": cfg.training.amp,
        "topography_type": cfg.loss.topography_type,
        "topology": cfg.loss.topology,
        "neighbourhood_type": cfg.loss.neighbourhood.type,
        "neighbourhood_radius": cfg.loss.neighbourhood.radius,
        "embedding_dim": cfg.model.embedding_dim,
        "p_dropout": cfg.model.p_dropout,
        "head_bias": cfg.model.head.bias,
        "model_arch": cfg.model.arch,
        "dataset": cfg.dataset.name,
        "transforms_preset": cfg.dataset.transforms.preset,
        "split_strategy": cfg.dataset.split.strategy,
        "val_per_class": cfg.dataset.split.val_per_class,
        "save_freq_epochs": cfg.training.save_freq_epochs,
        "early_stopping_patience": cfg.training.early_stopping_patience,
        "early_stopping_method": cfg.training.early_stopping_method,
        "beta": cfg.training.balancer.beta,
        "eps": cfg.training.balancer.eps,
        "lambda_max": cfg.training.balancer.lambda_max,
        # FFCV full-recipe params
        "loading_backend": cfg.training.loading_backend,
        "label_smoothing": cfg.training.label_smoothing,
        "use_blurpool": cfg.training.use_blurpool,
        "optimizer_selective_wd": cfg.training.optimizer_selective_wd,
        "lr_tta": cfg.training.lr_tta,
        "lr_peak_epoch": cfg.training.lr_peak_epoch,
        "progressive_res_min": cfg.training.progressive_res_min,
        "progressive_res_max": cfg.training.progressive_res_max,
        "progressive_res_start_ramp": cfg.training.progressive_res_start_ramp,
        "progressive_res_end_ramp": cfg.training.progressive_res_end_ramp,
        # FFCV beton format settings (None for torch runs)
        "beton_max_resolution": cfg.training.beton.max_resolution,
        "beton_jpeg_quality": cfg.training.beton.jpeg_quality,
        "beton_compress_probability": cfg.training.beton.compress_probability,
    }
    return {k: str(v) for k, v in raw.items() if v is not None}


# ── MLflow filter from dry_filter.* overrides ─────────────────────────────────

def _build_filter(cfg: DictConfig) -> tuple[str, bool]:
    clauses = ["tags.kind = 'model'", "attributes.status = 'FINISHED'"]
    has_filters = False
    dry = OmegaConf.to_container(cfg.dry_filter, resolve=True) if hasattr(cfg, "dry_filter") else {}
    if dry:
        for k, v in sorted(dry.items()):
            if v is not None:
                has_filters = True
                clauses.append(f"params.{k} = '{v}'")
    return " and ".join(clauses), has_filters


# ── Diff printing ─────────────────────────────────────────────────────────────

def _print_diff(
    run_id: str,
    score: int,
    total: int,
    proposed: dict[str, str],
    existing: dict[str, str],
    identity_fields: set[str],
    proposed_identity_hash: str,
    existing_identity_hash: str | None,
) -> None:
    all_keys = sorted(set(proposed.keys()) | set(existing.keys()))
    diff_rows = []
    for k in all_keys:
        p_val = proposed.get(k, "<missing>")
        e_val = existing.get(k, "<missing>")
        if p_val != e_val:
            marker = "*" if _is_identity_param(k, identity_fields) else ""
            diff_rows.append((f"{k}{marker}", p_val, e_val))

    # ── Stale-hash detection ──────────────────────────────────────────────────
    # All logged params match but the stored identity_hash differs.
    # This means the run's hash is outdated — the fix is a rehash migration,
    # NOT retraining.
    is_stale_hash = (
        not diff_rows
        and existing_identity_hash is not None
        and existing_identity_hash != proposed_identity_hash
    )

    if is_stale_hash:
        print(f"!!! run {run_id[:12]} ({score}/{total} params match) — STALE IDENTITY HASH !!!")
        print(f"  All logged params match but identity_hash differs.")
        print(f"  Proposed : {proposed_identity_hash}")
        print(f"  Stored   : {existing_identity_hash}")
        print(f"  Fix      : uv run scripts/migrations/rehash_identities.py --apply")
    else:
        print(f"─── run {run_id[:12]} ({score}/{total} params match) ───")

        if not diff_rows:
            print(
                "  (no differing params — hash miss may be due to a param that is "
                "hashed but not individually logged, or a schema/version mismatch)"
            )
        else:
            # Dynamic column widths
            w_param = max(len(r[0]) for r in diff_rows)
            w_prop  = max(max(len(r[1]) for r in diff_rows), len("Proposed"))
            w_exist = max(max(len(r[2]) for r in diff_rows), len("Existing"))
            # Cap at 30 chars to avoid wrapping
            w_prop  = min(w_prop, 30)
            w_exist = min(w_exist, 30)

            print(f"  {'Param':<{w_param}}   {'Proposed':<{w_prop}}   {'Existing':<{w_exist}}")
            print(f"  {'─'*w_param}   {'─'*w_prop}   {'─'*w_exist}")
            for disp_k, p_val, e_val in diff_rows:
                if len(p_val) > w_prop:
                    p_val = p_val[:w_prop - 3] + "..."
                if len(e_val) > w_exist:
                    e_val = e_val[:w_exist - 3] + "..."
                print(f"  {disp_k:<{w_param}}   {p_val:<{w_prop}}   {e_val:<{w_exist}}")

    print()

    return is_stale_hash


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("Training Dry Run")
    print("=" * 60)

    # ── Config resolution — mirrors 01_train_models.py exactly ──
    validate_training_config(cfg)
    seed = resolve_seed(cfg)
    cfg.seed = seed
    hash_val = cfg_hash(cfg)
    setup_mlflow(cfg)
    configure_run_repository(cfg.mlflow.tracking_uri, cfg.mlflow.experiment_name)

    existing_run, model_identity_hash = find_finished_model_run(cfg, seed)

    print(f"  identity_hash : {model_identity_hash}")
    print(f"  cfg_hash      : {hash_val}")
    print(f"  trial         : {cfg.trial} / seed: {seed}")
    print()

    # ── Idempotency hit ──
    if existing_run is not None:
        start_time = "?"
        if existing_run.info.start_time:
            start_time = datetime.fromtimestamp(
                existing_run.info.start_time / 1000.0
            ).strftime("%Y-%m-%d %H:%M:%S")
        print("  WOULD SKIP — FINISHED run already exists:")
        print(f"    run_id    : {existing_run.info.run_id}")
        print(f"    started   : {start_time}")
        print("=" * 60)
        return

    # ── No match — find candidates ──
    print("  WOULD TRAIN — no existing run matches this identity hash.")
    print()

    proposed_params = _build_proposed_params(cfg, seed)
    identity_fields = set(model_identity_fields(cfg, seed).keys())

    filter_str, has_filters = _build_filter(cfg)
    runs_df = search_runs(filter_str, output_format="pandas")

    if runs_df is None or runs_df.empty:
        print("Searching for similar FINISHED model runs...  (0 found)")
        print("=" * 60)
        return

    print(f"Searching for similar FINISHED model runs...  ({len(runs_df)} found)")
    print()

    # Extract params and tags from MLflow pandas df
    param_cols = [c for c in runs_df.columns if c.startswith("params.")]
    tag_id_col = "tags.identity_hash" if "tags.identity_hash" in runs_df.columns else None

    scores: list[tuple[int, str, dict[str, str], str | None]] = []

    for _, row in runs_df.iterrows():
        run_id = row.get("run_id")
        if not run_id:
            continue

        existing_params: dict[str, str] = {}
        for col in param_cols:
            val = row[col]
            if pd.notna(val):
                key = col[7:]  # strip 'params.'
                existing_params[key] = str(val)

        # Skip runs missing more than half the expected keys (likely schema-mismatched)
        overlap = sum(1 for k in proposed_params if k in existing_params)
        if overlap < len(proposed_params) / 2:
            continue

        score = sum(
            1 for k, v in proposed_params.items()
            if existing_params.get(k) == v
        )

        existing_id_hash: str | None = None
        if tag_id_col is not None:
            val = row.get(tag_id_col)
            if pd.notna(val):
                existing_id_hash = str(val)

        scores.append((score, run_id, existing_params, existing_id_hash))

    scores.sort(key=lambda x: x[0], reverse=True)

    if not has_filters and len(scores) > 5:
        print("  (No +dry_filter.* specified — limiting output to the top 5 closest matches)")
        print("  (Append +dry_filter.<param>=<value> to narrow the search instead)\n")
        scores = scores[:5]

    found_stale = False
    for score, run_id, existing_params, existing_id_hash in scores:
        is_stale = _print_diff(
            run_id=run_id,
            score=score,
            total=len(proposed_params),
            proposed=proposed_params,
            existing=existing_params,
            identity_fields=identity_fields,
            proposed_identity_hash=model_identity_hash,
            existing_identity_hash=existing_id_hash,
        )
        if is_stale:
            found_stale = True

    if any(score < len(proposed_params) for score, *_ in scores):
        print("  * Parameter is part of the identity hash.")

    print("=" * 60)


if __name__ == "__main__":
    main()
