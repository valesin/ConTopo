from __future__ import annotations

from typing import TypedDict

import torch


class InferenceOutput(TypedDict):
    preds: torch.Tensor
    labels: torch.Tensor
    logits: torch.Tensor
    probs: torch.Tensor
    embeddings: torch.Tensor
    accuracy: float


class AnchorSpec(TypedDict):
    source_split: str
    per_class: int
    strategy: str
    order_by: str
    num_classes: int


class AnchorOutput(TypedDict):
    anchor_indices: list[int]
    anchor_labels: torch.Tensor
    spec: AnchorSpec
    spec_hash: str
