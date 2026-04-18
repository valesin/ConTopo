"""
Inference runner.

``run_combined_model_inference`` runs a forward pass over a DataLoader and
returns embeddings, logits, predictions, probabilities, and accuracy.

Artifacts per run (logged to MLflow):
  - logits, preds, probs, embeddings, labels (as .npz)
  - accuracy is logged as an MLflow *metric*, not an artifact
"""

from __future__ import annotations

from collections.abc import Sized
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.types import InferenceOutput


@torch.no_grad()
def run_combined_model_inference(
    model: nn.Module,
    loader: DataLoader,
    device: Any,
) -> InferenceOutput:
    """
    Run inference on a model that returns ``(embeddings, logits)`` or just ``logits``.
    """
    model.eval()

    total = len(loader.dataset if isinstance(loader.dataset, Sized) else loader)
    preds_buf: torch.Tensor | None = None
    labels_buf: torch.Tensor | None = None
    emb_list: list[torch.Tensor] = []
    logits_buf: torch.Tensor | None = None
    offset = 0

    for batch in loader:
        images, labs = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        labs = labs.to(device, non_blocking=True)
        bs = labs.size(0)

        out = model(images)
        # Expects (embeddings, logits) — fails loudly if model output shape is wrong.
        embeddings_batch, logits = out[0], out[1]

        batch_preds = logits.argmax(dim=1)
        logits_cpu = logits.detach().cpu()
        batch_preds_cpu = batch_preds.cpu()
        labs_cpu = labs.cpu()

        if logits_buf is None:
            logits_buf = logits_cpu.new_empty(total, logits_cpu.size(1))
            preds_buf = batch_preds_cpu.new_empty(total)
            labels_buf = labs_cpu.new_empty(total)

        if logits_buf is None or preds_buf is None or labels_buf is None:
            raise RuntimeError("Inference buffers failed to initialize")

        preds_buf[offset : offset + bs] = batch_preds_cpu
        labels_buf[offset : offset + bs] = labs_cpu
        logits_buf[offset : offset + bs] = logits_cpu
        emb_list.append(embeddings_batch.detach().cpu())
        offset += bs

    if logits_buf is None or preds_buf is None or labels_buf is None:
        raise RuntimeError(
            "Inference loader produced no batches; cannot build outputs."
        )

    embeddings_all = emb_list[0].new_empty((total, emb_list[0].size(1)))
    emb_offset = 0
    for emb in emb_list:
        emb_bs = emb.size(0)
        embeddings_all[emb_offset : emb_offset + emb_bs] = emb
        emb_offset += emb_bs

    probs = logits_buf.softmax(dim=1)
    acc = float((preds_buf == labels_buf).float().mean().item())

    return {
        "preds": preds_buf,
        "labels": labels_buf,
        "logits": logits_buf,
        "probs": probs,
        "embeddings": embeddings_all,
        "accuracy": acc,
    }
