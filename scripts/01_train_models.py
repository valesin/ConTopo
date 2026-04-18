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
import warnings
import hydra
import mlflow
from mlflow.models.signature import infer_signature
from numpy.exceptions import VisibleDeprecationWarning
import torch

# Suppress torchvision's NumPy 2.4 deprecation warning in CIFAR dataset loading
warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig
from torch.amp import GradScaler

from src.config.hash import cfg_hash
from src.config.validation import validate_training_config
from src.data.loaders import (
    get_dataset_loaders,
    get_split_labels,
    shutdown_dataloader_workers,
)
from src.losses.balancer import GradNormBalancer
from src.losses.topographic import Global_Topographic_Loss, Local_WS_Loss
from src.mlflow_utils import (
    log_resolved_config,
    model_tags,
    setup_mlflow,
    resolve_seed,
    set_torch_seed,
    resolve_device,
    log_dataset_lineage,
)
from src.repositories.functional_run_repository import (
    configure_run_repository,
    find_finished_model_run,
)
from src.networks.registry import build_model, unwrap
from src.training.train_ce import train_one_epoch, validate
from src.mlflow_schema_logger import (
    log_params as schema_log_params,
    start_run as schema_start_run,
    timed_log_metrics,
    timed_log_metric,
    timed_log_model,
)


def _build_optimiser(cfg: DictConfig, model):
    name = cfg.training.optimiser.lower()
    lr = cfg.training.learning_rate
    wd = cfg.training.weight_decay

    # Selective weight decay: apply WD only to non-BN, non-bias params (FFCV recipe)
    if cfg.training.optimizer_selective_wd:
        no_wd_names = ("bn", "bias")
        wd_params = [
            p
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_wd_names)
        ]
        no_wd_params = [
            p for n, p in model.named_parameters() if any(nd in n for nd in no_wd_names)
        ]
        param_groups = [
            {"params": wd_params},
            {"params": no_wd_params, "weight_decay": 0.0},
        ]
    else:
        param_groups = model.parameters()

    if name == "adam":
        return torch.optim.Adam(param_groups, lr=lr, weight_decay=wd)
    elif name == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)
    elif name == "sgd":
        return torch.optim.SGD(
            param_groups,
            lr=lr,
            weight_decay=wd,
            momentum=cfg.training.momentum,
        )
    else:
        raise ValueError(f"Unknown optimiser: {name}")


def _build_scheduler(cfg: DictConfig, optimiser, steps_per_epoch: int):
    """Return an LR scheduler or None (when scheduler=none)."""
    name = cfg.training.scheduler.lower()
    if name == "cyclic":
        epochs = cfg.training.epochs
        # pct_start must be in (0, 1); clamp lr_peak_epoch to valid range
        pct_start = min(cfg.training.lr_peak_epoch, epochs - 1) / epochs
        return torch.optim.lr_scheduler.OneCycleLR(
            optimiser,
            max_lr=cfg.training.learning_rate,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
        )
    elif name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=cfg.training.epochs
        )
    return None  # "none"


def _resolve_loader_for_epoch(
    train_loaders, epoch: int, total_epochs: int, cfg: DictConfig
):
    """Return the appropriate train loader for this epoch.

    If ``train_loaders`` is a list (progressive resolution), selects based on the
    current epoch's position in the ramp schedule:
      - epoch < start_ramp * total → first loader (lowest resolution)
      - epoch >= end_ramp * total  → last loader (highest resolution)
      - in between                 → interpolates linearly across the list

    If ``train_loaders`` is a single loader, returns it unchanged.
    """
    if not isinstance(train_loaders, list):
        return train_loaders

    n = len(train_loaders)
    if n == 1:
        return train_loaders[0]

    start_frac = cfg.training.progressive_res_start_ramp
    end_frac = cfg.training.progressive_res_end_ramp
    frac = (epoch - 1) / total_epochs  # 0-based

    if frac < start_frac:
        return train_loaders[0]
    if frac >= end_frac:
        return train_loaders[-1]
    # Linear interpolation within the ramp
    ramp_frac = (frac - start_frac) / max(end_frac - start_frac, 1e-8)
    idx = min(int(ramp_frac * (n - 1)), n - 2)
    return train_loaders[idx]


def _validate_with_tta(model, loader, loss_fn, device, use_amp: bool):
    """Validate with test-time augmentation: average logits over original and HFlip.

    Returns (avg_loss, accuracy_fraction) matching the contract of ``validate``.
    Only called when ``lr_tta=True`` and ``loading_backend=ffcv``.
    """
    import torch.nn.functional as F

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).squeeze()

            # Forward pass 1: original
            out = model(images)
            logits1 = out[1] if isinstance(out, (tuple, list)) else out

            # Forward pass 2: horizontal flip
            out2 = model(images.flip(-1))
            logits2 = out2[1] if isinstance(out2, (tuple, list)) else out2

            logits = (logits1.float() + logits2.float()) * 0.5
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total_correct += logits.argmax(1).eq(labels).sum().item()
            total_n += labels.size(0)

    model.train()
    return total_loss / total_n, total_correct / total_n


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
    # ── Config validation ──
    validate_training_config(cfg)

    # ── Seed ──
    seed = resolve_seed(cfg)
    cfg.seed = seed
    set_torch_seed(seed)
    cudnn.benchmark = True

    # ── Idempotency ──
    hash_val = cfg_hash(cfg)
    setup_mlflow(cfg)
    configure_run_repository(cfg.mlflow.tracking_uri, cfg.mlflow.experiment_name)

    existing_run, model_identity_hash = find_finished_model_run(cfg, seed)
    if existing_run is not None:
        print(
            "Model already FINISHED. "
            f"run_id={existing_run.info.run_id}, "
            f"identity_hash={model_identity_hash}, cfg_hash={hash_val}. Skipping."
        )
        return

    # ── Data ──
    train_loader, val_loader, test_loader = get_dataset_loaders(cfg)

    try:
        # ── Device ──
        device = resolve_device(cfg.runtime.device)

        # ── Model ──
        model = build_model(cfg, ret_emb=True)
        model = model.to(device)

        # ── Data Parallel on multi-GPU setups ──
        if (
            cfg.runtime.data_parallel
            and device.type == "cuda"
            and torch.cuda.device_count() > 1
        ):
            model = torch.nn.DataParallel(model)

        # ── Blurpool (antialiased downsampling) ──
        if cfg.training.use_blurpool:
            from antialiased_cnns import BlurPool

            for name, module in unwrap(model).named_modules():
                if isinstance(module, torch.nn.MaxPool2d) and module.stride > 1:
                    parent, attr = name.rsplit(".", 1) if "." in name else ("", name)
                    parent_mod = (
                        unwrap(model)
                        if not parent
                        else dict(unwrap(model).named_modules())[parent]
                    )
                    setattr(
                        parent_mod,
                        attr,
                        BlurPool(module.kernel_size, stride=module.stride),
                    )

        # ── Losses ──
        label_smoothing = float(cfg.training.label_smoothing)
        task_loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(
            device
        )
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

        # ── Scheduler ──
        # steps_per_epoch: use first train loader for count (handles progressive list)
        _first_loader = (
            train_loader[0] if isinstance(train_loader, list) else train_loader
        )
        steps_per_epoch = len(_first_loader)
        scheduler = _build_scheduler(cfg, optimiser, steps_per_epoch)

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
                    # FFCV beton format settings (hash-included; None for torch runs)
                    "beton_max_resolution": cfg.training.beton.max_resolution,
                    "beton_jpeg_quality": cfg.training.beton.jpeg_quality,
                    "beton_compress_probability": cfg.training.beton.compress_probability,
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
                get_split_labels(cfg, "val"),
                "val",
                cfg.dataset.name,
                context="validation",
            )
            log_dataset_lineage(
                get_split_labels(cfg, "test"),
                "test",
                cfg.dataset.name,
                context="testing",
            )

            # ── Create Signature and Input Example (Before Loop) ──
            # Grab one batch from the train_loader (use first loader if progressive list)
            _sig_loader = (
                train_loader[0] if isinstance(train_loader, list) else train_loader
            )
            sig_inputs, _ = next(iter(_sig_loader))
            sig_inputs = sig_inputs.to(device).float()

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

            early_stopping_method = cfg.training.early_stopping_method.lower()
            if early_stopping_method not in ("val_acc", "val_loss"):
                raise ValueError(
                    f"Unknown early_stopping_method: {early_stopping_method}"
                )

            best_val_acc = 0.0
            best_val_loss = float("inf")
            best_epoch = 0
            epochs_no_improve = 0
            patience = cfg.training.early_stopping_patience

            # Initialize memory for the best model weights
            best_model_state = None

            use_tta = cfg.training.lr_tta and cfg.training.loading_backend == "ffcv"
            total_epochs = cfg.training.epochs
            sched_name = cfg.training.scheduler.lower()

            for epoch in range(1, total_epochs + 1):
                epoch_loader = _resolve_loader_for_epoch(
                    train_loader, epoch, total_epochs, cfg
                )
                metrics = train_one_epoch(
                    epoch_loader,
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
                    scheduler=scheduler if sched_name == "cyclic" else None,
                )

                # Per-epoch schedulers (not cyclic — cyclic steps per batch)
                if scheduler is not None and sched_name != "cyclic":
                    scheduler.step()

                if use_tta:
                    val_loss, val_acc = _validate_with_tta(
                        model, val_loader, task_loss_fn, device, use_amp
                    )
                else:
                    val_loss, val_acc = validate(
                        val_loader,
                        model,
                        task_loss_fn,
                        print_freq=cfg.runtime.print_freq,
                    )

                # ── Log metrics to MLflow ──
                timed_log_metrics(
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
                if cfg.training.save_checkpoints and epoch % save_freq == 0:
                    checkpoint_name = f"checkpoint_epoch{epoch:04d}"
                    timed_log_model(
                        unwrap(model),
                        name=checkpoint_name,
                        signature=signature,
                    )

                # ── Best checkpoint & early stopping ──
                # Check improvement before updating running bests
                improved = (
                    val_acc > best_val_acc
                    if early_stopping_method == "val_acc"
                    else val_loss < best_val_loss
                )

                # Always keep running best of each metric (for logging)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                if improved:
                    best_epoch = epoch
                    # Track best state IN MEMORY, avoid MLflow overwriting errors
                    best_model_state = copy.deepcopy(unwrap(model).state_dict())
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
            timed_log_metric("test_accuracy", test_acc)
            timed_log_metric("test_loss", test_loss)
            timed_log_metric("best_val_acc", best_val_acc)
            timed_log_metric("best_val_loss", best_val_loss)

            # ── Log the finalized best model artifact ONCE ──
            timed_log_model(
                unwrap(model),
                name="e2e_best",
                signature=signature,
            )

            print(f"Done. test_acc={test_acc:.4f}, run_id={run.info.run_id}")
    finally:
        if isinstance(train_loader, list):
            for tl in train_loader:
                shutdown_dataloader_workers(tl)
        else:
            shutdown_dataloader_workers(train_loader)
        shutdown_dataloader_workers(val_loader)
        shutdown_dataloader_workers(test_loader)


if __name__ == "__main__":
    main()
