"""Pure-NumPy sandbox dynamics (parity with ``attractor_llm.core`` + norm band [0.5, 3.0]).

This module is intentionally self-contained for fast tests and notebooks.
"""

from __future__ import annotations

import hashlib
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

FloatVector: TypeAlias = npt.NDArray[np.float64]
FloatMatrix: TypeAlias = npt.NDArray[np.float64]


def _pcg64_from_seed(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def text_to_signal(text: str, dim: int) -> FloatVector:
    """Deterministic unit-norm signal (same construction as ``attractor_llm.core``)."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    sseed = int.from_bytes(digest[:8], "big")
    rng = _pcg64_from_seed(sseed)
    v = rng.standard_normal(dim).astype(np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.ones(dim, dtype=np.float64) / np.sqrt(dim)
    return v / n


def make_diffusion_matrix(dim: int, rng: np.random.Generator) -> FloatMatrix:
    q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    eigenvalues = -0.2 - rng.uniform(0.05, 0.35, size=dim)
    return (q * eigenvalues) @ q.T


def center(state: FloatVector) -> FloatVector:
    return state - np.mean(state)


def step_state(
    state: FloatVector,
    diffusion: FloatMatrix,
    applied_signal: FloatVector,
    dt: float,
    cubic_scale: float = 0.05,
) -> FloatVector:
    c = center(state)
    nonlinear = cubic_scale * (c**3)
    drift = diffusion @ state + nonlinear + applied_signal
    return state + dt * drift


def stabilize_norm(
    v: FloatVector,
    floor: float = 0.5,
    ceiling: float = 3.0,
    *,
    value_clip: float | None = 5.0,
) -> FloatVector:
    """Elementwise clip + norm band (training-style), per vector."""
    out = np.asarray(v, dtype=np.float64)
    if value_clip is not None and value_clip > 0:
        out = np.clip(out, -value_clip, value_clip)
    n = np.linalg.norm(out)
    if floor <= 0 and ceiling <= 0:
        return out
    target = float(np.clip(n, floor, ceiling))
    if n < 1e-15:
        return out
    return out * (target / n)


def integrate(
    diffusion: FloatMatrix,
    signal: FloatVector,
    *,
    dim: int,
    num_steps: int = 24,
    dt: float = 0.05,
    cubic_scale: float = 0.05,
    norm_floor: float = 0.5,
    norm_ceiling: float = 3.0,
    value_clip: float | None = 5.0,
) -> FloatVector:
    """Fixed-step Euler with stabilization each step (matches training intent)."""
    s: FloatVector = np.zeros(dim, dtype=np.float64)
    for _ in range(num_steps):
        s = step_state(s, diffusion, signal, dt, cubic_scale=cubic_scale)
        s = stabilize_norm(s, norm_floor, norm_ceiling, value_clip=value_clip)
    return s


def proto_logits(state: FloatVector, attractors: FloatMatrix, tau: float = 1.0) -> FloatVector:
    """Negative-distance logits: ell_v = -||s - a_v|| / tau."""
    dist = np.linalg.norm(attractors - state, axis=1)
    return -dist / tau


def toy_cycle_story(
    vocab_size: int = 50,
    length: int = 120,
    seed: int = 0,
) -> list[int]:
    """Tiny synthetic token stream (cycle motifs)."""
    rng = np.random.default_rng(seed)
    motifs = [
        [10, 11, 12, 13],
        [20, 21, 22, 23],
        [30, 31, 32, 33],
    ]
    out: list[int] = []
    for _ in range(length):
        m = motifs[rng.integers(0, len(motifs))]
        out.append(int(m[rng.integers(0, len(m))] % max(vocab_size, 1)))
    return out


def save_toy_checkpoint(path: str, *, dim: int, seed: int) -> None:
    """Persist a tiny numpy checkpoint (npz) for demos."""
    rng = np.random.default_rng(seed)
    diff = make_diffusion_matrix(dim, rng)
    sig = text_to_signal("toy", dim)
    state = integrate(diff, sig, dim=dim, num_steps=16)
    np.savez_compressed(path, dim=dim, seed=seed, diffusion=diff, final_state=state)


def load_toy_checkpoint(path: str) -> dict[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=False))
