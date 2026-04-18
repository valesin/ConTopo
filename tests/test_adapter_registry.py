from __future__ import annotations

import pytest
from src.networks.adapter_registry import (
    ADAPTER_REGISTRY,
    build_adapter,
    adapter_architecture_name,
)


def test_all_meta_types_in_registry():
    """Builds every adapter contained in the factory registry dict."""
    for meta_type in ADAPTER_REGISTRY.keys():
        model = build_adapter(meta_type=meta_type, input_dim=10, num_classes=5)
        assert model is not None
        assert isinstance(model, ADAPTER_REGISTRY[meta_type])


def test_unknown_meta_type_raises():
    """Test standard unknown network factory fallback."""
    with pytest.raises(ValueError, match="Unknown meta_type 'unknown_xyz'"):
        build_adapter(meta_type="unknown_xyz", input_dim=123, num_classes=3)


def test_architecture_name_matches():
    """Verify simple architectural class logging names mapping."""
    for meta_type, cls in ADAPTER_REGISTRY.items():
        assert adapter_architecture_name(meta_type) == cls.__name__

    assert adapter_architecture_name("this_is_not_an_adapter") == "UnknownAdapter"
