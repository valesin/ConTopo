"""
Inference runner.

``run_combined_model_inference`` runs a forward pass over a DataLoader and
returns embeddings, logits, predictions, probabilities, and accuracy.

Artifacts per run (logged to MLflow):
  - logits, preds, probs, embeddings, labels (as .npz)
  - accuracy is logged as an MLflow *metric*, not an artifact
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def run_combined_model_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Run inference on a model that returns ``(embeddings, logits)`` or just ``logits``.
    """
    model.eval()

    total = len(loader.dataset)
    preds_buf = torch.empty(total, dtype=torch.long)
    labels_buf = torch.empty(total, dtype=torch.long)
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

        if logits_buf is None:
            logits_buf = torch.empty(total, logits_cpu.size(1), dtype=logits_cpu.dtype)

        preds_buf[offset : offset + bs] = batch_preds.cpu()
        labels_buf[offset : offset + bs] = labs.cpu()
        logits_buf[offset : offset + bs] = logits_cpu
        emb_list.append(embeddings_batch.detach().cpu())
        offset += bs

    if logits_buf is None:
        logits_buf = torch.empty(total, 0)

    embeddings_all = torch.cat(emb_list, dim=0)
    probs = torch.softmax(logits_buf, dim=1)
    acc = float((preds_buf == labels_buf).float().mean().item())

    return {
        "preds": preds_buf,
        "labels": labels_buf,
        "logits": logits_buf,
        "probs": probs,
        "embeddings": embeddings_all,
        "accuracy": acc,
    }
