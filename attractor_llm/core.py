"""State dynamics, deterministic signals, and convergence to stable attractors."""

from __future__ import annotations

import hashlib
from typing import Tuple

import numpy as np


def _pcg64_from_seed(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def text_to_signal(text: str, dim: int) -> np.ndarray:
    """
    Map text to a unit-norm signal vector without learned embeddings.
    Deterministic: same string always yields the same vector.
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


def make_diffusion_matrix(dim: int, rng: np.random.Generator) -> np.ndarray:
    """
    Negative-definite linear operator so unforced dynamics decay toward a stable basin.
    """
    q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    eigenvalues = -0.2 - rng.uniform(0.05, 0.35, size=dim)
    return (q * eigenvalues) @ q.T


def linear_diffusion(matrix: np.ndarray, state: np.ndarray) -> np.ndarray:
    return matrix @ state


def center(state: np.ndarray) -> np.ndarray:
    return state - np.mean(state)


def step_state(
    state: np.ndarray,
    diffusion: np.ndarray,
    applied_signal: np.ndarray,
    dt: float,
    cubic_scale: float = 0.05,
) -> np.ndarray:
    """
    state[t+1] = state[t] + Δt * (
        linear_diffusion(state[t]) + cubic_scale * center(state[t])^3 + applied_signal
    )
    """
    c = center(state)
    nonlinear = cubic_scale * (c**3)
    drift = linear_diffusion(diffusion, state) + nonlinear + applied_signal
    return state + dt * drift


def _clamp_norm(v: np.ndarray, floor: float, ceiling: float | None) -> np.ndarray:
    """Keep vectors non-zero and optionally bounded during integration."""
    n = np.linalg.norm(v)
    if n < floor:
        return v * (floor / (n + 1e-15))
    if ceiling is not None and n > ceiling:
        return v * (ceiling / n)
    return v


def converge(
    diffusion: np.ndarray,
    applied_signal: np.ndarray,
    dim: int,
    dt: float = 0.05,
    tol: float = 1e-4,
    max_steps: int = 50_000,
    initial_state: np.ndarray | None = None,
    cubic_scale: float = 0.05,
    magnitude_floor: float = 1e-3,
    magnitude_ceiling: float | None = 12.0,
    target_norm: float | None = None,
) -> Tuple[np.ndarray, int, float]:
    """
    Run dynamics until ||Δstate|| < tol or max_steps.
    Magnitude is kept in (floor, ceiling) during integration; if target_norm is set,
    the final state is rescaled to that norm (non-zero, stable output magnitude).
    Returns (final_state, steps, final_delta_norm).
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


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))
