"""PyTorch dynamics wrappers (parity with ``attractor_llm.torch_core``)."""

from __future__ import annotations

from attractor_llm.torch_core import (
    AttractorDynamics,
    MultiHeadDynamics,
    center,
    converge_fixed_steps,
    linear_diffusion,
    make_diffusion_matrix,
    step_state,
    text_to_signal,
)

__all__ = [
    "AttractorDynamics",
    "MultiHeadDynamics",
    "center",
    "converge_fixed_steps",
    "linear_diffusion",
    "make_diffusion_matrix",
    "step_state",
    "text_to_signal",
]
