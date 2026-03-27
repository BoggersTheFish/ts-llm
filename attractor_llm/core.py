"""NumPy attractor dynamics and deterministic signal primitives.

Note:
    This is the legacy reference implementation for attractor-state evolution.
    Any mathematical changes here must be treated as architecture changes.
"""

from __future__ import annotations

import hashlib
from typing import Tuple

import numpy as np
import numpy.typing as npt

FloatVector = npt.NDArray[np.float64]
FloatMatrix = npt.NDArray[np.float64]


def _pcg64_from_seed(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def text_to_signal(text: str, dim: int) -> FloatVector:
    """Map text to a deterministic unit-norm signal vector.

    Args:
        text: Input text seed.
        dim: Vector dimension.

    Returns:
        Unit-norm signal vector of length ``dim``.

    Note:
        Hash-based seeding ensures identical text produces identical vectors.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = _pcg64_from_seed(seed)
    v = rng.standard_normal(dim).astype(np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        v = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
    else:
        v = v / n
    return v


def make_diffusion_matrix(dim: int, rng: np.random.Generator) -> FloatMatrix:
    """Construct a negative-definite diffusion matrix.

    Args:
        dim: Matrix dimension.
        rng: NumPy random generator.

    Returns:
        ``(dim, dim)`` diffusion matrix with negative spectrum.

    Note:
        Negative-definite structure encourages stable unforced decay.
    """
    q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    eigenvalues = -0.2 - rng.uniform(0.05, 0.35, size=dim)
    return (q * eigenvalues) @ q.T


def linear_diffusion(matrix: FloatMatrix, state: FloatVector) -> FloatVector:
    return matrix @ state


def center(state: FloatVector) -> FloatVector:
    return state - np.mean(state)


def step_state(
    state: FloatVector,
    diffusion: FloatMatrix,
    applied_signal: FloatVector,
    dt: float,
    cubic_scale: float = 0.05,
) -> FloatVector:
    """Advance the state by one explicit Euler step.

    Args:
        state: Current state vector.
        diffusion: Diffusion matrix.
        applied_signal: External signal injection.
        dt: Euler timestep.
        cubic_scale: Cubic nonlinearity coefficient.

    Returns:
        Next state vector.

    Note:
        This is the canonical legacy attractor update rule.
    """
    c = center(state)
    nonlinear = cubic_scale * (c**3)
    drift = linear_diffusion(diffusion, state) + nonlinear + applied_signal
    return state + dt * drift


def _clamp_norm(v: FloatVector, floor: float, ceiling: float | None) -> FloatVector:
    """Keep vectors non-zero and optionally bounded during integration."""
    n = np.linalg.norm(v)
    if n < floor:
        return v * (floor / (n + 1e-15))
    if ceiling is not None and n > ceiling:
        return v * (ceiling / n)
    return v


def converge(
    diffusion: FloatMatrix,
    applied_signal: FloatVector,
    dim: int,
    dt: float = 0.05,
    tol: float = 1e-4,
    max_steps: int = 50_000,
    initial_state: FloatVector | None = None,
    cubic_scale: float = 0.05,
    magnitude_floor: float = 1e-3,
    magnitude_ceiling: float | None = 12.0,
    target_norm: float | None = None,
) -> Tuple[FloatVector, int, float]:
    """Integrate until convergence threshold or step budget is reached.

    Args:
        diffusion: Diffusion matrix.
        applied_signal: External signal injection vector.
        dim: State dimension.
        dt: Euler timestep.
        tol: Delta-norm convergence threshold.
        max_steps: Maximum Euler steps.
        initial_state: Optional initial state.
        cubic_scale: Cubic nonlinearity coefficient.
        magnitude_floor: Minimum norm clamp.
        magnitude_ceiling: Optional maximum norm clamp.
        target_norm: Optional final-state norm target.

    Returns:
        Tuple of ``(final_state, steps_used, final_delta_norm)``.

    Note:
        Clamp/target-norm behavior is preserved for parity with existing checkpoints.
    """
    if initial_state is None:
        s = np.zeros(dim, dtype=np.float64)
    else:
        s = np.array(initial_state, dtype=np.float64, copy=True)

    last_delta = float("inf")
    for k in range(max_steps):
        s_next = step_state(s, diffusion, applied_signal, dt, cubic_scale=cubic_scale)
        delta = np.linalg.norm(s_next - s)
        last_delta = delta
        s = _clamp_norm(s_next, magnitude_floor, magnitude_ceiling)
        if delta < tol:
            if target_norm is not None:
                n = np.linalg.norm(s)
                if n > 1e-12:
                    s = s * (target_norm / n)
                else:
                    s = np.ones(dim, dtype=np.float64) * (target_norm / np.sqrt(dim))
            return s, k + 1, last_delta
    if target_norm is not None:
        n = np.linalg.norm(s)
        if n > 1e-12:
            s = s * (target_norm / n)
        else:
            s = np.ones(dim, dtype=np.float64) * (target_norm / np.sqrt(dim))
    return s, max_steps, last_delta


def euclidean_distance(a: FloatVector, b: FloatVector) -> float:
    return float(np.linalg.norm(a - b))
