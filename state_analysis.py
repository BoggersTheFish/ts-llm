"""
Distance / similarity structure in prefix attractor states (NumPy only).

`states` is a dict prefix -> 1D float array from `collect_prefix_states` in eval_harness.
Matrices use rows/cols in sorted key order (deterministic).
"""

from __future__ import annotations

import numpy as np


def _stack_sorted(states: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    labels = sorted(states.keys())
    X = np.stack([states[k] for k in labels], axis=0).astype(np.float64, copy=False)
    return X, labels


def pairwise_cosine(states: dict[str, np.ndarray]) -> np.ndarray:
    """
    Pairwise cosine *distance* matrix: 1 - cosine_similarity.
    Shape (n, n); order of rows/columns is sorted(prefix keys).
    """
    X, _ = _stack_sorted(states)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    Xn = X / norms
    sim = Xn @ Xn.T
    return (1.0 - sim).astype(np.float64)


def pairwise_euclidean(states: dict[str, np.ndarray]) -> np.ndarray:
    """Pairwise L2 distance matrix; same key order as `pairwise_cosine`."""
    X, _ = _stack_sorted(states)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def labels_sorted(states: dict[str, np.ndarray]) -> list[str]:
    """Axis labels matching matrix row/column order."""
    return sorted(states.keys())
