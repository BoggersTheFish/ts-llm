"""
PyTorch attractor dynamics — differentiable under fixed-step integration.

This module mirrors :mod:`attractor_llm.core` (NumPy) with the same mathematical
structure. Training should use :func:`converge_fixed_steps` (fixed Euler steps, no
early stopping) so the number of operations is constant and autograd is well-defined.
Inference may use :func:`converge_adaptive` to match the legacy adaptive loop.

**Dynamics (explicit Euler step).** Let :math:`s_t \\in \\mathbb{R}^D` be the state,
:math:`A \\in \\mathbb{R}^{D \\times D}` the diffusion operator, :math:`\\alpha` the
cubic coefficient, :math:`c(s) = s - \\bar{s}` with :math:`\\bar{s}` the mean of
:math:`s` along the last dimension (per-vector, batch-safe), and :math:`u` the applied
signal. One step is

.. math::

    s_{t+1} = s_t + \\Delta t\\,\\bigl( A s_t + \\alpha\\, c(s_t)^{\\odot 3} + u \\bigr),

where :math:`(\\cdot)^{\\odot 3}` is the element-wise cube.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from attractor_llm.embeddings import LearnableProtoEmbedder


def text_to_signal(
    text: str,
    dim: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Deterministic unit-norm signal (same construction as NumPy :func:`attractor_llm.core.text_to_signal`).

    Uses SHA-256 of ``text`` to seed a torch :class:`~torch.Generator`, then draws
    :math:`v \\sim \\mathcal{N}(0, I)` and returns :math:`v / \\|v\\|_2` (or uniform
    direction if the norm is degenerate).
    """
    if device is None:
        device = torch.device("cpu")
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    v = torch.randn(dim, generator=g, device=device, dtype=dtype)
    n = torch.linalg.vector_norm(v)
    if n <= 1e-12:
        return torch.ones(dim, device=device, dtype=dtype) / (dim**0.5)
    return v / n


def make_diffusion_matrix(
    dim: int,
    rng: torch.Generator | None = None,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Negative-definite linear operator :math:`A = Q \\operatorname{diag}(\\lambda) Q^\\top`
    with :math:`\\lambda_i \\in [-0.55, -0.25]` (same sampling as NumPy: ``-0.2 - U(0.05, 0.35)``).

    Becomes a learnable parameter when wrapped in :class:`AttractorDynamics`.
    """
    if device is None:
        device = torch.device("cpu")
    if rng is None:
        rng = torch.Generator(device=device)
        rng.manual_seed(42)
    q = torch.linalg.qr(torch.randn(dim, dim, generator=rng, device=device, dtype=dtype))[0]
    u = torch.rand(dim, generator=rng, device=device, dtype=dtype)
    eigenvalues = -0.2 - (0.05 + 0.3 * u)
    return (q * eigenvalues) @ q.T


def linear_diffusion(matrix: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """
    Apply diffusion :math:`A s`. Supports ``state`` shape ``(D,)`` or ``(..., D)`` via
    :math:`s A^\\top` (row-vector convention).
    """
    if state.ndim == 1:
        return matrix @ state
    return state @ matrix.T


def center(state: torch.Tensor) -> torch.Tensor:
    """
    Centering :math:`c(s)`. For a 1D tensor (D,), uses a scalar mean (NumPy parity).
    For ``(..., D)``, centers each last-dimensional vector independently so batches do
    not mix.
    """
    if state.ndim == 1:
        return state - state.mean()
    return state - state.mean(dim=-1, keepdim=True)


def step_state(
    state: torch.Tensor,
    diffusion: torch.Tensor,
    applied_signal: torch.Tensor,
    dt: float | torch.Tensor,
    *,
    cubic_scale: float | torch.Tensor = 0.05,
) -> torch.Tensor:
    r"""
    One explicit Euler step:

    .. math::

        s_{t+1} = s_t + \Delta t\,\bigl( A s_t + \alpha\, c(s_t)^{\odot 3} + u \bigr).
    """
    c = center(state)
    alpha = torch.as_tensor(cubic_scale, device=state.device, dtype=state.dtype)
    nonlinear = alpha * (c**3)
    drift = linear_diffusion(diffusion, state) + nonlinear + applied_signal
    dt_t = torch.as_tensor(dt, device=state.device, dtype=state.dtype)
    return state + dt_t * drift


def _clamp_norm(
    v: torch.Tensor,
    floor: float,
    ceiling: float | None,
) -> torch.Tensor:
    """Keep norms in ``[floor, ceiling]`` per last dimension (batch-safe)."""
    n = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    out = v
    if floor > 0:
        out = torch.where(n < floor, out * (floor / (n + 1e-15)), out)
        n = torch.linalg.vector_norm(out, dim=-1, keepdim=True)
    if ceiling is not None:
        out = torch.where(n > ceiling, out * (ceiling / n), out)
    return out


def _apply_target_norm(s: torch.Tensor, dim: int, target_norm: float | None) -> torch.Tensor:
    if target_norm is None:
        return s
    n = torch.linalg.vector_norm(s, dim=-1, keepdim=True)
    if s.ndim == 1:
        if n.item() > 1e-12:
            return s * (target_norm / n)
        return torch.ones(dim, device=s.device, dtype=s.dtype) * (target_norm / (dim**0.5))
    return torch.where(
        n > 1e-12,
        s * (target_norm / n),
        torch.ones_like(s) * (target_norm / (dim**0.5)),
    )


def converge_adaptive(
    diffusion: torch.Tensor,
    applied_signal: torch.Tensor,
    dim: int,
    dt: float = 0.05,
    tol: float = 1e-4,
    max_steps: int = 50_000,
    initial_state: torch.Tensor | None = None,
    cubic_scale: float | torch.Tensor = 0.05,
    magnitude_floor: float = 1e-3,
    magnitude_ceiling: float | None = 12.0,
    target_norm: float | None = 1.0,
) -> Tuple[torch.Tensor, int, float]:
    """
    Adaptive convergence (inference-style): stop when incremental norm ``< tol`` or
    ``max_steps`` reached. Matches :func:`attractor_llm.core.converge` numerically (same
    structure). Not differentiable in the sense of variable iteration count.
    """
    if initial_state is None:
        s = torch.zeros(dim, device=applied_signal.device, dtype=applied_signal.dtype)
    else:
        s = initial_state.clone()

    last_delta = float("inf")
    for k in range(max_steps):
        s_next = step_state(s, diffusion, applied_signal, dt, cubic_scale=cubic_scale)
        delta = float(torch.linalg.vector_norm(s_next - s).item())
        last_delta = delta
        s = _clamp_norm(s_next, magnitude_floor, magnitude_ceiling)
        if delta < tol:
            s = _apply_target_norm(s, dim, target_norm)
            return s, k + 1, last_delta
    s = _apply_target_norm(s, dim, target_norm)
    return s, max_steps, last_delta


def converge_fixed_steps(
    diffusion: torch.Tensor,
    applied_signal: torch.Tensor,
    dim: int,
    num_steps: int,
    dt: float = 0.05,
    initial_state: torch.Tensor | None = None,
    cubic_scale: float | torch.Tensor = 0.05,
    magnitude_floor: float = 1e-3,
    magnitude_ceiling: float | None = 12.0,
    target_norm: float | None = 1.0,
) -> torch.Tensor:
    """
    Fixed ``num_steps`` Euler integration — **fully differentiable** w.r.t. parameters
    in ``diffusion``, ``cubic_scale``, ``applied_signal``, and ``initial_state`` (no
    early exit). Uses the same per-step clamp and optional final ``target_norm``
    rescaling as the NumPy path.

    If ``applied_signal`` has shape ``(V, D)`` with :math:`V > 1`, integrates **each row**
    in parallel (same :math:`A,\\alpha`), returning ``(V, D)`` attractors.
    """
    if applied_signal.ndim == 2:
        v, d = applied_signal.shape
        if d != dim:
            raise ValueError("applied_signal last dim must match state dim")
        if initial_state is None:
            s = torch.zeros(v, d, device=applied_signal.device, dtype=applied_signal.dtype)
        else:
            s = initial_state.clone()
        for _ in range(num_steps):
            s_next = step_state(s, diffusion, applied_signal, dt, cubic_scale=cubic_scale)
            s = _clamp_norm(s_next, magnitude_floor, magnitude_ceiling)
        return _apply_target_norm(s, dim, target_norm)

    if initial_state is None:
        s = torch.zeros(dim, device=applied_signal.device, dtype=applied_signal.dtype)
    else:
        s = initial_state.clone()

    for _ in range(num_steps):
        s_next = step_state(s, diffusion, applied_signal, dt, cubic_scale=cubic_scale)
        s = _clamp_norm(s_next, magnitude_floor, magnitude_ceiling)

    return _apply_target_norm(s, dim, target_norm)


def euclidean_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Euclidean distance :math:`\\|a - b\\|_2` (returns a 0-d tensor)."""
    return torch.linalg.vector_norm(a - b)


class AttractorDynamics(nn.Module):
    """
    Learnable vector field: :math:`A` (diffusion) and :math:`\\alpha` (cubic coefficient)
    as parameters; fixed :math:`\\Delta t` by default.

    Use :meth:`forward` for one differentiable Euler step, :meth:`converge_fixed` for
    training, and :meth:`adaptive_converge` for inference aligned with the legacy loop.
    """

    def __init__(
        self,
        dim: int = 128,
        diffusion: torch.Tensor | None = None,
        *,
        cubic_scale: float = 0.05,
        dt: float = 0.05,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self.dim = dim
        self.dt = dt
        self.cubic_scale = nn.Parameter(torch.tensor(cubic_scale, device=device, dtype=dtype))
        diff = diffusion if diffusion is not None else make_diffusion_matrix(dim, device=device, dtype=dtype)
        self.diffusion = nn.Parameter(diff)
        self.to(device=device, dtype=dtype)

    def center(self, state: torch.Tensor) -> torch.Tensor:
        return center(state)

    def forward(self, state: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
        """One Euler step; differentiable."""
        return step_state(state, self.diffusion, signal, self.dt, cubic_scale=self.cubic_scale)

    def converge_fixed(
        self,
        signal: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        *,
        num_steps: int = 20,
        magnitude_floor: float = 1e-3,
        magnitude_ceiling: float | None = 12.0,
        target_norm: float | None = 1.0,
    ) -> torch.Tensor:
        """
        Fixed-step convergence for training (see :func:`converge_fixed_steps`).
        Supports ``signal`` of shape ``(D,)`` or ``(V, D)`` for batched attractors.
        """
        return converge_fixed_steps(
            self.diffusion,
            signal,
            self.dim,
            num_steps,
            dt=self.dt,
            initial_state=initial_state,
            cubic_scale=self.cubic_scale,
            magnitude_floor=magnitude_floor,
            magnitude_ceiling=magnitude_ceiling,
            target_norm=target_norm,
        )

    def adaptive_converge(
        self,
        signal: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        *,
        tol: float = 1e-4,
        max_steps: int = 50_000,
        magnitude_floor: float = 1e-3,
        magnitude_ceiling: float | None = 12.0,
        target_norm: float | None = 1.0,
    ) -> Tuple[torch.Tensor, int, float]:
        """Adaptive convergence for inference (see :func:`converge_adaptive`)."""
        return converge_adaptive(
            self.diffusion,
            signal,
            self.dim,
            dt=self.dt,
            tol=tol,
            max_steps=max_steps,
            initial_state=initial_state,
            cubic_scale=self.cubic_scale,
            magnitude_floor=magnitude_floor,
            magnitude_ceiling=magnitude_ceiling,
            target_norm=target_norm,
        )

    def precompute_attractors(
        self,
        embedder: LearnableProtoEmbedder,
        num_steps: int = 16,
    ) -> torch.Tensor:
        """
        For each vocabulary token :math:`v`, compute an attractor by holding fixed the
        corresponding signal :math:`u_v` and integrating from :math:`s_0 = 0` with
        :meth:`converge_fixed` (differentiable w.r.t. ``embedder`` and this module).

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(V, D)`` where row :math:`v` is the converged state for
            token :math:`v`.
        """
        signals = embedder.get_all_signals()
        if signals.ndim != 2:
            raise ValueError("embedder.get_all_signals must return (V, D)")
        return self.converge_fixed(signals, num_steps=num_steps, initial_state=None)
