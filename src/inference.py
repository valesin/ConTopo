"""
Inference runner + caching layer.

``get_or_run_inference`` is the single entry-point:
  1. Check for local cache (per-split artifacts)
  2. If missing, run inference and save artifacts
  3. Return loaded tensors

Artifacts per run (logged to MLflow and saved locally):
  - logits.pt, preds.pt, probs.pt, embeddings.pt
  - labels.pt, hashes.pt, original_indices.pt
  - accuracy is logged as an MLflow *metric*, not an artifact
"""

from __future__ import annotations

import os
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.cache import StorageBackend, get_backend


# ───────────── inference runner ─────────────


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
        if isinstance(out, (tuple, list)):
            embeddings_batch, logits = out[0], out[-1]
        else:
            logits = out
            embeddings_batch = out  # if no separate embeddings

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


# ───────────── artifact I/O ─────────────

ARTIFACT_KEYS = ["logits", "preds", "probs", "embeddings", "labels", "hashes", "original_indices"]


def save_inference_artifacts(
    data: Dict[str, Any],
    artifact_dir: str,
    backend: StorageBackend | None = None,
) -> None:
    """Save each artifact key as a separate file."""
    if backend is None:
        backend = get_backend("pt")
    os.makedirs(artifact_dir, exist_ok=True)
    for key in ARTIFACT_KEYS:
        if key in data and data[key] is not None:
            path = os.path.join(artifact_dir, f"{key}{backend.extension}")
            backend.save(data[key], path)


def load_inference_artifacts(
    artifact_dir: str,
    backend: StorageBackend | None = None,
) -> Dict[str, Any]:
    """Load all available artifact files from a directory."""
    if backend is None:
        backend = get_backend("pt")
    result: Dict[str, Any] = {}
    for key in ARTIFACT_KEYS:
        path = os.path.join(artifact_dir, f"{key}{backend.extension}")
        if backend.exists(path):
            result[key] = backend.load(path)
    return result


def artifacts_complete(artifact_dir: str, backend: StorageBackend | None = None) -> bool:
    """Check that the minimum required artifacts exist."""
    if backend is None:
        backend = get_backend("pt")
    required = ["logits", "preds", "labels"]
    return all(
        backend.exists(os.path.join(artifact_dir, f"{k}{backend.extension}"))
        for k in required
    )


# ───────────── main API ─────────────


def get_or_run_inference(
    *,
    model: nn.Module | None = None,
    loader: DataLoader | None = None,
    device: torch.device | None = None,
    artifact_dir: str,
    manifest_data: Dict[str, Any] | None = None,
    backend_name: str = "pt",
    force: bool = False,
) -> Dict[str, Any]:
    """
    Get cached inference or run it.

    If cached artifacts exist in ``artifact_dir`` and ``force=False``, load and return.
    Otherwise, run inference using ``model`` + ``loader`` and save.

    ``manifest_data`` should contain ``hashes`` and ``original_indices`` that
    will be included in the saved artifacts.
    """
    backend = get_backend(backend_name)

    if not force and artifacts_complete(artifact_dir, backend):
        data = load_inference_artifacts(artifact_dir, backend)
        return data

    if model is None or loader is None:
        raise ValueError(
            "Cached artifacts not found and model/loader not provided. "
            f"artifact_dir={artifact_dir}"
        )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = run_combined_model_inference(model, loader, device)

    # Attach manifest alignment data if provided
    if manifest_data is not None:
        results["hashes"] = manifest_data.get("hashes")
        results["original_indices"] = manifest_data.get("original_indices")
        # labels from manifest take precedence (ground truth)
        if "labels" in manifest_data:
            results["labels"] = manifest_data["labels"]

    save_inference_artifacts(results, artifact_dir, backend)
    return results
