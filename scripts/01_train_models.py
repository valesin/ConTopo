#!/usr/bin/env python3
"""
01_train_models.py — Hydra + MLflow CE training entrypoint.

Usage:
    python scripts/01_train_models.py loss.rho=0.05 trial=0
    python scripts/01_train_models.py loss.topology=grid loss.rho=0.04 trial=2
    python scripts/01_train_models.py --multirun \\
        loss.rho=0,0.008,0.04,0.2,1,5 \\
        loss.topology=torus,grid \\
        trial=0,1,2,3,4
"""

from __future__ import annotations

import os

import hydra
import mlflow
import torch
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler

from src.config.hash import cfg_hash
from src.config.paths import get_cache_dir, get_models_dir
from src.config.schema import apply_schema_defaults
from src.data.loaders import get_cifar10_loaders
from src.data.manifest import get_or_create_manifest
from src.losses.balancer import GradNormBalancer
from src.losses.topographic import Global_Topographic_Loss, Local_WS_Loss
from src.mlflow_utils import find_finished_run, log_git_info, log_resolved_config, model_tags, setup_mlflow
from src.networks.registry import build_model, to_device, unwrap
from src.training.checkpoint import save_checkpoint
from src.training.train_ce import train_one_epoch, validate


def _resolve_seed(cfg: DictConfig) -> int:
    if cfg.seed is not None:
        return int(cfg.seed)
    return 100 + int(cfg.trial)


def _build_optimizer(cfg: DictConfig, model):
    name = cfg.training.optimizer.lower()
    lr = cfg.training.learning_rate
    wd = cfg.training.weight_decay
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=wd,
            momentum=cfg.training.momentum,
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def _build_topo_loss(cfg: DictConfig, emb_dim: int):
    topo_type = cfg.loss.topography_type
    if topo_type == "ws":
        return Local_WS_Loss(weight=1.0, topology=cfg.loss.topology)
    elif topo_type == "global":
        return Global_Topographic_Loss(weight=1.0, emb_dim=emb_dim)
    else:
        raise ValueError(f"Unknown topography_type: {topo_type}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = apply_schema_defaults(cfg)

    # ── Seed ──
    seed = _resolve_seed(cfg)
    cfg.seed = seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True

    # ── Idempotency ──
    hash_val = cfg_hash(cfg)
    setup_mlflow(cfg)

    existing = find_finished_run(cfg.mlflow.experiment_name, hash_val, kind="model")
    if existing is not None:
        print(f"Run with cfg_hash={hash_val} already FINISHED (run_id={existing.info.run_id}). Skipping.")
        return

    # ── Data ──
    train_loader, val_loader, test_loader = get_cifar10_loaders(cfg)

    # ── Dataset manifest (test split) ──
    manifest = get_or_create_manifest(
        dataset_name=cfg.dataset.name,
        split="test",
        data_root=cfg.runtime.data_root,
        artifacts_root=str(get_cache_dir(cfg)),
    )
    manifest_hash = manifest.manifest_hash

    # ── Device ──
    dev_name = cfg.runtime.device
    if dev_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev_name)
    print(f"Using device: {device}")

    # ── Model ──
    model = build_model(cfg, ret_emb=True)
    model = model.to(device)
    if cfg.runtime.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # ── Losses ──
    task_loss_fn = torch.nn.CrossEntropyLoss().to(device)
    topo_loss_fn = _build_topo_loss(cfg, cfg.model.embedding_dim)
    if isinstance(topo_loss_fn, torch.nn.Module):
        topo_loss_fn = topo_loss_fn.to(device)

    # ── Balancer ──
    balancer = GradNormBalancer(
        rho=float(cfg.loss.rho),
        beta=cfg.training.balancer.beta,
        eps=cfg.training.balancer.eps,
        lambda_max=cfg.training.balancer.lambda_max,
    )

    # ── Optimizer ──
    optimizer = _build_optimizer(cfg, model)

    # ── AMP ──
    use_amp = bool(cfg.training.amp)
    scaler = GradScaler("cuda", enabled=use_amp) if use_amp else None

    # ── Model save directory ──
    rho_str = str(float(cfg.loss.rho))
    topo_str = cfg.loss.topology
    trial_str = f"trial_{int(cfg.trial):02d}"
    model_dir = str(get_models_dir(cfg) / f"CE_{topo_str}_rho{rho_str}" / trial_str)
    os.makedirs(model_dir, exist_ok=True)

    save_freq = max(1, int(cfg.training.save_freq_epochs))

    # ── MLflow run ──
    tags = model_tags(cfg, hash_val, dataset_manifest_hash=manifest_hash)

    with mlflow.start_run(run_name=f"CE_{topo_str}_rho{rho_str}/{trial_str}", tags=tags) as run:
        # Log experiment-semantic params
        mlflow.log_params({
            "schema_version": cfg.schema_version,
            "rho": float(cfg.loss.rho),
            "trial": cfg.trial,
            "seed": seed,
            "epochs": cfg.training.epochs,
            "batch_size": cfg.training.batch_size,
            "learning_rate": cfg.training.learning_rate,
            "optimizer": cfg.training.optimizer,
            "weight_decay": cfg.training.weight_decay,
            "momentum": cfg.training.momentum,
            "scheduler": cfg.training.scheduler,
            "amp": cfg.training.amp,
            "topography_type": cfg.loss.topography_type,
            "topology": cfg.loss.topology,
            "neighborhood_type": cfg.loss.neighborhood.type,
            "neighborhood_radius": cfg.loss.neighborhood.radius,
            "embedding_dim": cfg.model.embedding_dim,
            "p_dropout": cfg.model.p_dropout,
            "head_bias": cfg.model.head.bias,
            "model_arch": cfg.model.arch,
            "dataset": cfg.dataset.name,
            "transforms_preset": cfg.dataset.transforms.preset,
            "split_strategy": cfg.dataset.split.strategy,
            "split_seed": cfg.dataset.split.seed,
            "val_per_class": cfg.dataset.split.val_per_class,
            "dataset_manifest_hash": manifest_hash,
        })
        log_git_info()

        # Log resolved config as artifact
        log_resolved_config(cfg)

        best_val_acc = 0.0
        epochs_no_improve = 0
        patience = cfg.training.early_stopping_patience

        for epoch in range(1, cfg.training.epochs + 1):
            prev_best = best_val_acc

            metrics = train_one_epoch(
                train_loader, model, task_loss_fn, topo_loss_fn, optimizer, epoch, balancer,
                topography_type=cfg.loss.topography_type,
                print_freq=cfg.runtime.print_freq,
                use_amp=use_amp,
                scaler=scaler,
            )

            val_loss, val_acc = validate(
                val_loader, model, task_loss_fn,
                print_freq=cfg.runtime.print_freq,
            )

            # ── Log metrics to MLflow ──
            mlflow.log_metrics({
                "train_total_loss": metrics["total_loss"],
                "train_task_loss": metrics["task_loss"],
                "train_topo_loss": metrics["topo_loss"],
                "lambda_hat": metrics["lambda_hat"],
                "train_acc": metrics["train_acc"],
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, step=epoch)

            # ── Periodic checkpoint ──
            if epoch % save_freq == 0:
                state = {
                    "stage": "e2e",
                    "epoch": epoch,
                    "state_dict": unwrap(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metrics": {**metrics, "val_loss": val_loss, "val_acc": val_acc},
                }
                ckpt_path = os.path.join(model_dir, f"e2e_epoch{epoch:04d}.pth")
                save_checkpoint(ckpt_path, state)

            # ── Best checkpoint ──
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {
                    "stage": "e2e",
                    "epoch": epoch,
                    "state_dict": unwrap(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": OmegaConf.to_container(cfg, resolve=True),
                    "metrics": {**metrics, "val_loss": val_loss, "val_acc": val_acc},
                }
                best_path = os.path.join(model_dir, "e2e_best.pth")
                save_checkpoint(best_path, best_state)

            # ── Early stopping ──
            if best_val_acc > prev_best:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        # ── Final test evaluation ──
        best_path = os.path.join(model_dir, "e2e_best.pth")
        if os.path.isfile(best_path):
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            unwrap(model).load_state_dict(ckpt["state_dict"])
        test_loss, test_acc = validate(
            test_loader, model, task_loss_fn,
            print_freq=cfg.runtime.print_freq,
        )
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("best_val_acc", best_val_acc)

        # ── Log checkpoint artifact ──
        if os.path.isfile(best_path):
            mlflow.log_artifact(best_path, artifact_path="checkpoint")

        print(f"Done. test_acc={test_acc:.4f}, run_id={run.info.run_id}")


if __name__ == "__main__":
    main()
