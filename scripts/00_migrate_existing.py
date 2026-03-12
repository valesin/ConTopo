#!/usr/bin/env python3
"""
00_migrate_existing.py — Migrate legacy CE checkpoints into MLflow.

Scans ``save/ResNet18/models/CE_rho*/trial_*/e2e_best.pth``, creates an
MLflow run for each, logs params/tags/artifacts, and ensures idempotency.

Usage:
    python scripts/00_migrate_existing.py
    python scripts/00_migrate_existing.py migration.dry_run=true
    python scripts/00_migrate_existing.py runtime.models_root=save/ResNet18old/models
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import tempfile

import hydra
import mlflow
import torch
from omegaconf import DictConfig

from src.mlflow_utils import log_git_info, log_resolved_config, setup_mlflow


# ───────────── helpers ─────────────


def _parse_model_dir_name(name: str) -> dict:
    """Extract rho, topography_type, topology, etc. from legacy folder name."""
    info: dict = {"model_dir_name": name}
    # e.g. crossentropy_wstopo_torus_256embdims_0rho_125epochs_512bsz_2nwork_0.002lr_0.5dropout
    # or   CE_rho0.04  (simplified alias-style name)
    # Try simplified pattern first
    # New naming convention: CE_<topology>_rho<value>  (e.g. CE_torus_rho0.04)
    m2 = re.match(r"CE_(torus|grid)_rho([\d.]+)", name)
    if m2:
        info["topology"] = m2.group(1)
        info["rho"] = float(m2.group(2))
        info["loss_type"] = "cross_entropy"
        info["topography_type"] = "ws"
        info["embedding_dim"] = 256
        info["epochs"] = 200
        info["batch_size"] = 512
        info["learning_rate"] = 0.002
        info["p_dropout"] = 0.5
        return info

    # Simplified legacy pattern: CE_rho<value>
    m = re.match(r"CE_rho([\d.]+)", name)
    if m:
        info["rho"] = float(m.group(1))
        info["loss_type"] = "cross_entropy"
        # Defaults for the simplified names
        info["topography_type"] = "ws"
        info["topology"] = "torus"
        info["embedding_dim"] = 256
        info["epochs"] = 125
        info["batch_size"] = 512
        info["learning_rate"] = 0.002
        info["p_dropout"] = 0.5
        return info

    # Full legacy pattern
    info["loss_type"] = "cross_entropy"
    patterns = {
        "rho":           r"([\d.]+)rho",
        "epochs":        r"(\d+)epochs",
        "batch_size":    r"(\d+)bsz",
        "learning_rate": r"([\d.]+)lr",
        "p_dropout":     r"([\d.]+)dropout",
        "embedding_dim": r"(\d+)embdims",
    }
    for key, pat in patterns.items():
        match = re.search(pat, name)
        if match:
            val = match.group(1)
            info[key] = float(val) if "." in val else int(val)

    if "wstopo" in name:
        info["topography_type"] = "ws"
    elif "globaltopo" in name:
        info["topography_type"] = "global"

    if "_torus_" in name or "torus" in name:
        info["topology"] = "torus"
    elif "_grid_" in name or "grid" in name:
        info["topology"] = "grid"
    else:
        info["topology"] = "torus"  # default for legacy CE runs

    return info


def _trial_index(trial_name: str) -> int:
    m = re.search(r"(\d+)", trial_name)
    return int(m.group(1)) if m else 0


def _build_cfg_hash_for_legacy(params: dict) -> str:
    """
    Build a deterministic cfg_hash for a legacy run.

    Uses the same canonical-JSON approach as src.mlflow_utils.cfg_hash,
    but constructed from extracted params.
    """
    canonical = json.dumps(params, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _already_migrated(experiment_name: str, cfg_hash_val: str) -> bool:
    """Check whether a run with this cfg_hash already exists."""
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return False
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.cfg_hash = '{cfg_hash_val}' and attributes.status = 'FINISHED'",
        max_results=1,
        output_format="list",
    )
    return len(runs) > 0


# ───────────── main ─────────────


def migrate(cfg: DictConfig) -> None:
    """Core migration logic, called from Hydra main."""
    models_root = cfg.runtime.models_root
    experiment_name = cfg.mlflow.experiment_name
    dry_run = cfg.migration.dry_run

    # Discover all legacy CE trial directories
    pattern = os.path.join(models_root, "CE_rho*", "trial_*", "e2e_best.pth")
    # Full legacy name pattern
    pattern2 = os.path.join(models_root, "crossentropy_*", "trial_*", "e2e_best.pth")
    # New naming convention with topology: CE_torus_rho*, CE_grid_rho*
    pattern3 = os.path.join(models_root, "CE_torus_rho*", "trial_*", "e2e_best.pth")
    pattern4 = os.path.join(models_root, "CE_grid_rho*", "trial_*", "e2e_best.pth")
    ckpt_paths = sorted(set(
        glob.glob(pattern) + glob.glob(pattern2) +
        glob.glob(pattern3) + glob.glob(pattern4)
    ))

    if not ckpt_paths:
        print(f"No legacy checkpoints found matching {pattern} or {pattern2}")
        return

    print(f"Found {len(ckpt_paths)} legacy checkpoints to migrate.")

    migrated = 0
    skipped = 0
    for ckpt_path in ckpt_paths:
        trial_dir = os.path.dirname(ckpt_path)
        model_dir = os.path.dirname(trial_dir)
        model_dir_name = os.path.basename(model_dir)
        trial_name = os.path.basename(trial_dir)
        trial_idx = _trial_index(trial_name)

        info = _parse_model_dir_name(model_dir_name)
        rho = info.get("rho", 0.0)
        seed = 100 + trial_idx

        # Build reproducible params dict for cfg_hash
        params = {
            "model_arch": "LinearResNet18",
            "embedding_dim": info.get("embedding_dim", 256),
            "num_classes": 10,
            "use_dropout": True,
            "p_dropout": info.get("p_dropout", 0.5),
            "loss_type": "cross_entropy",
            "topography_type": info.get("topography_type", "ws"),
            "topology": info.get("topology", "torus"),
            "rho": rho,
            "trial": trial_idx,
            "seed": seed,
            "epochs": info.get("epochs", 125),
            "batch_size": info.get("batch_size", 512),
            "learning_rate": info.get("learning_rate", 0.002),
            "dataset": "cifar10",
        }
        cfg_hash_val = _build_cfg_hash_for_legacy(params)

        if _already_migrated(experiment_name, cfg_hash_val):
            skipped += 1
            print(f"  SKIP (already migrated): {model_dir_name}/{trial_name}")
            continue

        if dry_run:
            print(f"  DRY-RUN: would migrate {model_dir_name}/{trial_name}  cfg_hash={cfg_hash_val}")
            continue

        # Load checkpoint to extract metrics
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_args = ckpt.get("args", {})
        ckpt_metrics = ckpt.get("metrics", {})
        ckpt_epoch = ckpt.get("epoch", -1)

        with mlflow.start_run(run_name=f"{model_dir_name}/{trial_name}") as run:
            # ── Tags ──
            mlflow.set_tags({
                "kind": "model",
                "loss_type": "cross_entropy",
                "rho": str(rho),
                "trial": str(trial_idx),
                "topography_type": str(info.get("topography_type", "ws")),
                "topology": str(info.get("topology", "torus")),
                "dataset": "cifar10",
                "model_arch": "LinearResNet18",
                "cfg_hash": cfg_hash_val,
                "migrated": "true",
                "legacy_model_dir": model_dir_name,
                "legacy_trial": trial_name,
            })

            # ── Params ──
            mlflow.log_params(params)

            # ── Metrics from checkpoint ──
            if ckpt_metrics:
                for mk, mv in ckpt_metrics.items():
                    if isinstance(mv, (int, float)):
                        mlflow.log_metric(mk.replace("/", "_"), mv)
            if "val_acc" in ckpt:
                mlflow.log_metric("val_acc", ckpt["val_acc"])
            if ckpt_epoch > 0:
                mlflow.log_metric("best_epoch", ckpt_epoch)

            # ── Artifacts ──
            # Log checkpoint
            mlflow.log_artifact(ckpt_path, artifact_path="checkpoint")

            # Log inference cache if it exists
            inf_cache = os.path.join(trial_dir, "inference_cifar.pt")
            if os.path.isfile(inf_cache):
                # Load and re-save as separate artifacts for the new convention
                inf_data = torch.load(inf_cache, map_location="cpu", weights_only=False)
                with tempfile.TemporaryDirectory() as tmpdir:
                    for key in ["logits", "preds", "labels", "embeddings"]:
                        if key in inf_data and inf_data[key] is not None:
                            tmp_path = os.path.join(tmpdir, f"{key}.pt")
                            torch.save(inf_data[key], tmp_path)
                            mlflow.log_artifact(tmp_path, artifact_path="inference/test")
                    # Log probs if logits exist
                    if "logits" in inf_data and inf_data["logits"] is not None:
                        probs = torch.softmax(inf_data["logits"], dim=1)
                        tmp_path = os.path.join(tmpdir, "probs.pt")
                        torch.save(probs, tmp_path)
                        mlflow.log_artifact(tmp_path, artifact_path="inference/test")
                    # Log accuracy as metric
                    if "accuracy" in inf_data:
                        mlflow.log_metric("test_accuracy", inf_data["accuracy"])

            # Git info (best effort)
            log_git_info()
            log_resolved_config(cfg)

        migrated += 1
        print(f"  MIGRATED: {model_dir_name}/{trial_name}  run_id={run.info.run_id}")

    print(f"\nDone. Migrated: {migrated}, Skipped: {skipped}, Total: {len(ckpt_paths)}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_mlflow(cfg)
    migrate(cfg)


if __name__ == "__main__":
    main()
