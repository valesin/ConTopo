"""
Storage backend abstraction for inference artifacts.

Default backend: `.pt` (PyTorch serialisation).
Zarr backend: stub that raises ``NotImplementedError``; structure is ready.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict

import torch


class StorageBackend(ABC):
    """Minimal interface for reading / writing tensors or dicts."""

    @abstractmethod
    def save(self, data: Any, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> Any: ...

    @abstractmethod
    def exists(self, path: str) -> bool: ...

    @property
    @abstractmethod
    def extension(self) -> str: ...


class PtBackend(StorageBackend):
    """PyTorch .pt file backend."""

    def save(self, data: Any, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(data, path)

    def load(self, path: str) -> Any:
        return torch.load(path, weights_only=False)

    def exists(self, path: str) -> bool:
        return os.path.isfile(path)

    @property
    def extension(self) -> str:
        return ".pt"


class ZarrBackend(StorageBackend):
    """Zarr backend — structural stub, raises on use."""

    def save(self, data: Dict[str, Any], path: str) -> None:
        raise NotImplementedError(
            "Zarr backend is not yet implemented.  Install contopo[zarr] and implement."
        )

    def load(self, path: str) -> Dict[str, Any]:
        raise NotImplementedError("Zarr backend is not yet implemented.")

    def exists(self, path: str) -> bool:
        raise NotImplementedError("Zarr backend is not yet implemented.")

    @property
    def extension(self) -> str:
        return ".zarr"


_BACKENDS: dict[str, type[StorageBackend]] = {
    "pt": PtBackend,
    "zarr": ZarrBackend,
}


def get_backend(name: str = "pt") -> StorageBackend:
    """Factory — returns a backend instance by name."""
    cls = _BACKENDS.get(name)
    if cls is None:
        raise ValueError(f"Unknown storage backend '{name}'. Available: {list(_BACKENDS)}")
    return cls()
