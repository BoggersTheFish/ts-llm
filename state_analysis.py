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


def basin_separation_ratio(
    states: dict[str, np.ndarray],
    class_labels: dict[str, str],
) -> float:
    """
    Mean between-class cosine distance / mean within-class cosine distance.

    A ratio > 1 means different classes are further apart on average than
    same-class pairs — the geometry encodes class structure.
    Returns NaN if all classes are singletons (no within-class pairs exist).

    Args:
        states:       prefix → 1-D float array (same convention as pairwise_cosine)
        class_labels: prefix → class name string
    """
    sorted_keys = sorted(states.keys())
    n = len(sorted_keys)
    if n < 2:
        return float("nan")

    X, _ = _stack_sorted(states)
    classes = [class_labels.get(k, k) for k in sorted_keys]

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    Xn = X / norms
    cos_dist = 1.0 - (Xn @ Xn.T)

    within: list[float] = []
    between: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(cos_dist[i, j])
            if classes[i] == classes[j]:
                within.append(d)
            else:
                between.append(d)

    if not within or not between:
        return float("nan")

    return float(np.mean(between) / np.mean(within))


def intrinsic_dimensionality(states: dict[str, np.ndarray]) -> float:
    """
    Participation ratio of the covariance eigenspectrum.

    PR = (Σλ_i)² / Σλ_i²

    Interpretation:
      PR ≈ 1    all variance concentrated in one direction (collapsed)
      PR ≈ d    variance spread across d dimensions equally (full capacity used)

    Requires at least 2 samples; returns NaN otherwise.
    """
    X, _ = _stack_sorted(states)
    if X.shape[0] < 2:
        return float("nan")
    X = X.astype(np.float64) - X.mean(axis=0, keepdims=True)
    cov = (X.T @ X) / (X.shape[0] - 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0.0)  # clip numerical negatives
    sum_l = float(eigvals.sum())
    sum_l2 = float((eigvals ** 2).sum())
    if sum_l2 < 1e-30:
        return float("nan")
    return float(sum_l ** 2 / sum_l2)
