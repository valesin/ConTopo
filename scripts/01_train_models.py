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

import copy
import json

import hydra
import mlflow
from mlflow.models.signature import infer_signature
import torch
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler

from src.config.hash import cfg_hash, identity_hash
from src.data.loaders import get_cifar10_loaders, get_split_labels
from src.losses.balancer import GradNormBalancer
from src.losses.topographic import Global_Topographic_Loss, Local_WS_Loss
from src.mlflow_utils import (
    log_resolved_config,
    model_tags,
    setup_mlflow,
    check_existing_model,
    resolve_seed,
    log_dataset_lineage,
)
from src.networks.registry import build_model, unwrap
from src.training.train_ce import train_one_epoch, validate
from src.mlflow_schema_logger import (
    log_params as schema_log_params,
    start_run as schema_start_run,
)


def _build_optimiser(cfg: DictConfig, model):
    name = cfg.training.optimiser.lower()
    lr = cfg.training.learning_rate
    wd = cfg.training.weight_decay
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            momentum=cfg.training.momentum,
        )
    else:
        raise ValueError(f"Unknown optimiser: {name}")


def _build_topo_loss(cfg: DictConfig, emb_dim: int):
    topo_type = cfg.loss.topography_type
    if topo_type == "ws":
        return Local_WS_Loss(weight=1.0, topology=cfg.loss.topology)
    elif topo_type == "global":
        return Global_Topographic_Loss(weight=1.0, emb_dim=emb_dim)
    else:
        raise ValueError(f"Unknown topography_type: {topo_type}")


def _flatten_identity_section(prefix: str, section: DictConfig) -> dict[str, str]:
    """Flatten a config section to dot-path string fields for model identity hashing.

    ``None`` values are stringified as ``"None"`` to preserve deterministic identity input.
    """
    out: dict[str, str] = {}

    def _walk(node: object, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                _walk(v, f"{path}.{k}")
            return
        if isinstance(node, list):
            out[path] = json.dumps(node, sort_keys=True)
            return
        out[path] = str(node)

    _walk(OmegaConf.to_container(section, resolve=True), prefix)
    return out


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # ── Seed ──
    seed = resolve_seed(cfg)
    cfg.seed = seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True

    # ── Idempotency ──
    hash_val = cfg_hash(cfg)
    model_identity_hash = identity_hash(
        "model",
        schema_version=str(cfg.schema_version),
        trial=str(cfg.trial),
        seed=str(seed),
        **_flatten_identity_section("model", cfg.model),
        **_flatten_identity_section("loss", cfg.loss),
        **_flatten_identity_section("dataset", cfg.dataset),
        **_flatten_identity_section("training", cfg.training),
    )
    setup_mlflow(cfg)

    exists = check_existing_model(cfg.mlflow.experiment_name, hash_val, kind="model")
    if exists:
        print(f"Run with cfg_hash={hash_val} already FINISHED. Skipping.")
        return

    # ── Data ──
    train_loader, val_loader, test_loader = get_cifar10_loaders(cfg)

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
    if (
        cfg.runtime.data_parallel
        and device.type == "cuda"
        and torch.cuda.device_count() > 1
    ):
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

    # ── Optimiser ──
    optimiser = _build_optimiser(cfg, model)

    # ── AMP ──
    use_amp = bool(cfg.training.amp)
    scaler = GradScaler("cuda", enabled=use_amp) if use_amp else None

    # ── Setup Tracking Variables ──
    rho_str = str(float(cfg.loss.rho))
    topo_str = cfg.loss.topology
    trial_str = f"trial_{int(cfg.trial):02d}"
    save_freq = max(1, int(cfg.training.save_freq_epochs))

    # ── MLflow run ──
    tags = model_tags(cfg, hash_val)
    tags["identity_hash"] = model_identity_hash

    with schema_start_run(
        kind="model", run_name=f"CE_{topo_str}_rho{rho_str}/{trial_str}", tags=tags
    ) as run:
        # Log experiment-semantic params
        schema_log_params(
            "model",
            {
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
                "split_seed": cfg.dataset.split.seed,
                "val_per_class": cfg.dataset.split.val_per_class,
            },
        )
        log_resolved_config(cfg)

        log_dataset_lineage(
            get_split_labels(cfg, "train"),
            "train",
            cfg.dataset.name,
            context="training",
        )
        log_dataset_lineage(
            get_split_labels(cfg, "val"), "val", cfg.dataset.name, context="validation"
        )
        log_dataset_lineage(
            get_split_labels(cfg, "test"), "test", cfg.dataset.name, context="testing"
        )

        # ── Create Signature and Input Example (Before Loop) ──
        # Grab one batch from the train_loader
        sig_inputs, _ = next(iter(train_loader))
        sig_inputs = sig_inputs.to(device)

        # Isolate a single image/sample for the example
        input_sample = sig_inputs[:1]

        # Get the model output for this sample
        unwrap(model).eval()
        with torch.no_grad():
            output_sample = unwrap(model)(input_sample)

        # Convert PyTorch tensors to NumPy arrays for MLflow
        input_example_np = input_sample.cpu().numpy()

        # The model is built with ret_emb=True, so it returns (embeddings, logits)
        emb_sample, logit_sample = output_sample
        output_example_np = {
            "embeddings": emb_sample.cpu().numpy(),
            "logits": logit_sample.cpu().numpy(),
        }

        # Infer the signature once
        signature = infer_signature(input_example_np, output_example_np)

        # Set the model back to train mode before the loop begins
        unwrap(model).train()

        best_val_acc = 0.0
        best_epoch = 0
        epochs_no_improve = 0
        patience = cfg.training.early_stopping_patience

        # Initialize memory for the best model weights
        best_model_state = None

        for epoch in range(1, cfg.training.epochs + 1):
            prev_best = best_val_acc

            metrics = train_one_epoch(
                train_loader,
                model,
                task_loss_fn,
                topo_loss_fn,
                optimiser,
                epoch,
                balancer,
                topography_type=cfg.loss.topography_type,
                print_freq=cfg.runtime.print_freq,
                use_amp=use_amp,
                scaler=scaler,
            )

            val_loss, val_acc = validate(
                val_loader,
                model,
                task_loss_fn,
                print_freq=cfg.runtime.print_freq,
            )

            # ── Log metrics to MLflow ──
            mlflow.log_metrics(
                {
                    "train_total_loss": metrics["total_loss"],
                    "train_task_loss": metrics["task_loss"],
                    "train_topo_loss": metrics["topo_loss"],
                    "lambda_hat": metrics["lambda_hat"],
                    "train_acc": metrics["train_acc"],
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                step=epoch,
            )

            # ── Periodic checkpoint ──
            if epoch % save_freq == 0:
                checkpoint_name = f"checkpoint_epoch{epoch:04d}"
                mlflow.pytorch.log_model(
                    unwrap(model),
                    name=checkpoint_name,
                    signature=signature,
                )

            # ── Best checkpoint ──
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                # Track best state IN MEMORY, avoid MLflow overwriting errors
                best_model_state = copy.deepcopy(unwrap(model).state_dict())

            # ── Early stopping ──
            if best_val_acc > prev_best:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        # ── Final test evaluation ──
        # Load the best weights from memory back into the model
        if best_model_state is not None:
            unwrap(model).load_state_dict(best_model_state)

        test_loss, test_acc = validate(
            test_loader,
            model,
            task_loss_fn,
            print_freq=cfg.runtime.print_freq,
        )

        # Log final metrics
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("best_val_acc", best_val_acc)

        # ── Log the finalized best model artifact ONCE ──
        mlflow.pytorch.log_model(
            unwrap(model),
            name="e2e_best",
            signature=signature,
        )

        print(f"Done. test_acc={test_acc:.4f}, run_id={run.info.run_id}")


if __name__ == "__main__":
    main()
