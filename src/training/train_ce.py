"""
CE training loop with topographic loss + grad-norm balancing.

This module provides the per-epoch training function and the validation function
used by ``scripts/01_train_models.py``.

Supports optional AMP (automatic mixed precision) via ``use_amp`` parameter.
"""

from __future__ import annotations

import sys
import time
from typing import Tuple

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from src.losses.balancer import GradNormBalancer
from src.networks.registry import unwrap


# ───────────── helpers ─────────────


class AverageMeter:
    """Running average tracker."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)):
    """Top-k accuracy (%) for a batch."""
    with torch.no_grad():
        maxk = max(topk)
        bs = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / bs))
        return res


# ───────────── training ─────────────


def train_one_epoch(
    train_loader,
    model: nn.Module,
    task_loss_fn: nn.Module,
    topo_loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    balancer: GradNormBalancer,
    *,
    topography_type: str = "ws",
    print_freq: int = 10,
    use_amp: bool = False,
    scaler: GradScaler | None = None,
) -> dict:
    """
    Train for one epoch.  Returns a dict of averaged metrics.

    Keys: total_loss, task_loss, topo_loss, lambda_hat, train_acc.
    """
    model.train()

    meters = {k: AverageMeter() for k in ("loss", "task", "topo", "lam", "acc")}
    batch_time = AverageMeter()
    end = time.time()

    for idx, (images, labels) in enumerate(train_loader):
        device = next(model.parameters()).device
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        bsz = labels.shape[0]

        with autocast("cuda", enabled=use_amp):
            embeddings, logits = model(images)
            task_loss = task_loss_fn(logits, labels)

        # Topographic loss + measurement params for grad-norm
        # NOTE: GradNormBalancer uses torch.autograd.grad which requires
        # float32 loss. We compute topo_loss outside autocast when AMP is on.
        if topography_type == "ws":
            base = unwrap(model)
            linear_layer = base.encoder.fc
            topo_loss = topo_loss_fn(linear_layer=linear_layer)
            measure_params = list(linear_layer.parameters())
        elif topography_type == "global":
            topo_loss = topo_loss_fn(embeddings.float())
            base = unwrap(model)
            measure_params = [p for p in base.encoder.parameters() if p.requires_grad]
        else:
            topo_loss = torch.tensor(0.0, device=device)
            measure_params = [p for p in model.parameters() if p.requires_grad]

        scale = balancer.step(task_loss.float(), topo_loss.float(), measure_params)
        loss = task_loss + scale * topo_loss

        optimizer.zero_grad()
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        acc1 = accuracy(logits.float(), labels)[0]
        meters["loss"].update(loss.item(), bsz)
        meters["task"].update(task_loss.item(), bsz)
        meters["topo"].update(topo_loss.item(), bsz)
        meters["lam"].update(balancer.lambda_hat, bsz)
        meters["acc"].update(float(acc1), bsz)

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % print_freq == 0:
            print(
                f"Epoch [{epoch}][{idx+1}/{len(train_loader)}]  "
                f"Loss {meters['loss'].avg:.4f}  Task {meters['task'].avg:.4f}  "
                f"Topo {meters['topo'].avg:.4f}  λ {meters['lam'].avg:.4f}  "
                f"Acc {meters['acc'].avg:.2f}%  "
                f"Time {batch_time.avg:.3f}s"
            )
            sys.stdout.flush()

    return {
        "total_loss": meters["loss"].avg,
        "task_loss": meters["task"].avg,
        "topo_loss": meters["topo"].avg,
        "lambda_hat": meters["lam"].avg,
        "train_acc": meters["acc"].avg,
    }


# ───────────── validation ─────────────

class _LogitsOnly(nn.Module):
    """Adapter to make a (embeddings, logits) model return logits only."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out[1] if isinstance(out, (tuple, list)) else out


def validate(
    loader,
    model: nn.Module,
    loss_fn: nn.Module,
    *,
    print_freq: int = 10,
) -> Tuple[float, float]:
    """
    Evaluate on a loader.  Returns (avg_loss, accuracy_fraction).

    The model may return ``(emb, logits)`` or just ``logits``.
    """
    logits_model = _LogitsOnly(model)
    logits_model.eval()

    losses = AverageMeter()
    acc_meter = AverageMeter()
    device = next(model.parameters()).device

    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            bsz = labels.size(0)

            logits = logits_model(images)
            loss = loss_fn(logits, labels)
            losses.update(loss.item(), bsz)

            _, preds = logits.max(1)
            acc_meter.update(preds.eq(labels).float().mean().item(), bsz)

            if idx % print_freq == 0:
                print(
                    f"  Val [{idx}/{len(loader)}]  "
                    f"Loss {losses.avg:.4f}  Acc {acc_meter.avg:.4f}"
                )

    return losses.avg, acc_meter.avg
