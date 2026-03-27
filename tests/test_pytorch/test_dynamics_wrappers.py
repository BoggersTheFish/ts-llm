"""Tests for ``ts_attractor.dynamics`` re-exports."""

from __future__ import annotations

import pytest
import torch

from ts_attractor.dynamics import (
    AttractorDynamics,
    MultiHeadDynamics,
    center,
    make_diffusion_matrix,
    step_state,
    text_to_signal,
)


@pytest.mark.parametrize("dim", [8, 16, 32])
def test_make_diffusion_negative_definite(dim: int) -> None:
    m = make_diffusion_matrix(dim, device=torch.device("cpu"))
    e = torch.linalg.eigvalsh(m)
    assert bool(torch.all(e < 0))


def test_step_state_matches_dims() -> None:
    dim = 16
    d = make_diffusion_matrix(dim)
    s = torch.zeros(dim)
    u = text_to_signal("t", dim)
    s2 = step_state(s, d, u, 0.05)
    assert s2.shape == (dim,)


def test_center_batch() -> None:
    x = torch.randn(2, 8)
    c = center(x)
    assert c.shape == x.shape
    assert torch.allclose(c.mean(dim=-1), torch.zeros(2), atol=1e-6)


def test_attractor_dynamics_forward() -> None:
    dyn = AttractorDynamics(dim=12)
    s = torch.zeros(12)
    u = torch.randn(12)
    s2 = dyn.forward(s, u)
    assert s2.shape == (12,)


@pytest.mark.parametrize("heads", [2, 4])
def test_multihead_forward(heads: int) -> None:
    dim = 8 * heads
    m = MultiHeadDynamics(state_dim=dim, num_heads=heads, rank=4)
    s = torch.randn(dim)
    u = torch.randn(dim)
    out = m.forward(s, u)
    assert out.shape == (dim,)
