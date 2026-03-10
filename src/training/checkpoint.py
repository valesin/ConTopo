"""
Checkpoint save / load utilities.
"""

import os
from typing import Any, Dict

import torch
import torch.nn as nn


def save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, device: torch.device | str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)
