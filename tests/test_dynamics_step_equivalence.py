"""Dynamics consistency checks between NumPy and Torch step functions."""

from __future__ import annotations

import numpy as np
import torch

from attractor_llm.core import step_state as numpy_step_state
from attractor_llm.torch_core import step_state as torch_step_state


def test_euler_step_equivalence_numpy_vs_torch() -> None:
    """One Euler step should match between NumPy and Torch implementations."""
    rng = np.random.default_rng(123)
    dim = 8
    state = rng.standard_normal(dim).astype(np.float64)
    diffusion = rng.standard_normal((dim, dim)).astype(np.float64)
    signal = rng.standard_normal(dim).astype(np.float64)
    dt = 0.05
    cubic_scale = 0.07

    np_next = numpy_step_state(state, diffusion, signal, dt, cubic_scale=cubic_scale)
    t_next = torch_step_state(
        torch.tensor(state, dtype=torch.float64),
        torch.tensor(diffusion, dtype=torch.float64),
        torch.tensor(signal, dtype=torch.float64),
        dt,
        cubic_scale=cubic_scale,
    )
    np.testing.assert_allclose(np_next, t_next.detach().cpu().numpy(), rtol=1e-10, atol=1e-10)

