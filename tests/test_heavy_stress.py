"""CPU-heavy stress tests (skipped unless RUN_HEAVY_TESTS=1)."""

from __future__ import annotations

import pytest
import torch

from ts_attractor.dynamics import MultiHeadDynamics, text_to_signal


@pytest.mark.heavy
def test_8192_multistep_no_nan() -> None:
    dim = 8192
    heads = 8
    m = MultiHeadDynamics(state_dim=dim, num_heads=heads, rank=32)
    s = torch.zeros(dim)
    u = text_to_signal("stress", dim)
    for _ in range(100):
        s = m.forward(s, u)
    assert torch.isfinite(s).all()


@pytest.mark.heavy
def test_memory_stub_config_small_forward() -> None:
    """Placeholder for 1B-scale memory profiling; uses tiny forward only."""
    m = MultiHeadDynamics(state_dim=256, num_heads=8, rank=16)
    s = torch.randn(256)
    u = torch.randn(256)
    y = m.forward(s, u)
    assert y.shape == (256,)
