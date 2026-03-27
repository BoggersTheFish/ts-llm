#!/usr/bin/env python3
"""Generate toy checkpoints for 64 / 512 / 4096 dims.

For ``dim >= 2048`` we avoid constructing a full :math:`D\\times D` diffusion matrix
(expensive on laptops); the checkpoint stores a deterministic normalized state vector only.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from ts_attractor.numpy_demo import integrate, make_diffusion_matrix, text_to_signal


def _cheap_large_state(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    v = v / (np.linalg.norm(v) + 1e-12) * 1.5
    return v.astype(np.float64)


def main() -> None:
    out = REPO_ROOT / "checkpoints" / "toy"
    out.mkdir(parents=True, exist_ok=True)
    specs = [(64, 1), (512, 2), (4096, 3)]
    for dim, seed in specs:
        if dim >= 2048:
            state = _cheap_large_state(dim, seed)
            kind = "cheap_large"
        else:
            rng = np.random.default_rng(seed)
            diff = make_diffusion_matrix(dim, rng)
            sig = text_to_signal("cycle story toy", dim)
            state = integrate(diff, sig, dim=dim, num_steps=12)
            kind = "integrated"
        path = out / f"toy_dim_{dim}.npz"
        np.savez_compressed(path, dim=dim, seed=seed, final_state=state, kind=kind)
        print("wrote", path, kind)


if __name__ == "__main__":
    main()
