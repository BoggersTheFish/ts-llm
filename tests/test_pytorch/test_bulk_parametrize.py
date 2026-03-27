"""Bulk parametrized tests (fast, small tensors)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ts_attractor import numpy_demo as nd
from ts_attractor.dynamics import center, step_state, text_to_signal
from ts_attractor.dynamics import make_diffusion_matrix as make_d


@pytest.mark.parametrize("dim", [4, 8, 12, 16])
@pytest.mark.parametrize("dt", [0.01, 0.05, 0.1])
def test_numpy_step_deterministic(dim: int, dt: float) -> None:
    rng = np.random.default_rng(42)
    diff = nd.make_diffusion_matrix(dim, rng)
    sig = nd.text_to_signal("p", dim)
    s = np.zeros(dim)
    s2 = nd.step_state(s, diff, sig, dt)
    assert np.all(np.isfinite(s2))


@pytest.mark.parametrize("dim", [8, 16, 24, 32])
def test_torch_step_finite(dim: int) -> None:
    g = torch.Generator()
    g.manual_seed(99)
    d = make_d(dim, rng=g)
    s = torch.zeros(dim)
    u = text_to_signal("q", dim)
    s2 = step_state(s, d, u, 0.05)
    assert torch.isfinite(s2).all()


@pytest.mark.parametrize("b", [1, 2, 3, 4])
def test_center_batch_rows(b: int) -> None:
    x = torch.randn(b, 11)
    c = center(x)
    assert c.shape == x.shape


@pytest.mark.parametrize("floor,ceil", [(0.5, 2.0), (0.5, 3.0), (1.0, 2.5)])
def test_stabilize_norm(floor: float, ceil: float) -> None:
    v = np.random.default_rng(0).standard_normal(20) * 5
    out = nd.stabilize_norm(v, floor, ceil)
    n = float(np.linalg.norm(out))
    assert n <= ceil + 1e-5
    assert n >= floor - 1e-5 or np.linalg.norm(v) < floor


@pytest.mark.parametrize("tau", [0.5, 1.0, 2.0])
def test_proto_logits_tau(tau: float) -> None:
    s = np.zeros(5)
    att = np.eye(5)[:3]
    ell = nd.proto_logits(s, att, tau=tau)
    assert ell.shape == (3,)
