"""
Tests for state_analysis metrics introduced in Phase 2:
  - basin_separation_ratio
  - intrinsic_dimensionality
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from state_analysis import basin_separation_ratio, intrinsic_dimensionality, pairwise_cosine


# ---------------------------------------------------------------------------
# basin_separation_ratio
# ---------------------------------------------------------------------------

def _make_states(vecs: dict[str, list[float]]) -> dict[str, np.ndarray]:
    return {k: np.array(v, dtype=np.float64) for k, v in vecs.items()}


def test_bsr_well_separated_classes() -> None:
    """
    Two classes whose centroids are far apart and within-class variance is zero.
    Between-class distance >> 0; within-class distance = 0, so ratio = nan.
    Use non-zero within-class spread to get a finite ratio.
    """
    # Class A: vectors near [1, 0, 0]
    # Class B: vectors near [0, 1, 0]
    states = _make_states({
        "a1": [1.0, 0.0, 0.0],
        "a2": [0.9, 0.1, 0.0],
        "b1": [0.0, 1.0, 0.0],
        "b2": [0.1, 0.9, 0.0],
    })
    labels = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}
    ratio = basin_separation_ratio(states, labels)
    assert ratio == ratio, "ratio is NaN"
    assert ratio > 1.0, f"Expected ratio > 1 for well-separated classes, got {ratio:.4f}"


def test_bsr_identical_classes_gives_nan() -> None:
    """All singletons: no within-class pairs → NaN."""
    states = _make_states({"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.5, 0.5]})
    labels = {"a": "A", "b": "B", "c": "C"}
    ratio = basin_separation_ratio(states, labels)
    assert math.isnan(ratio), f"Expected NaN for all-singleton classes, got {ratio}"


def test_bsr_interleaved_classes_below_one() -> None:
    """
    Classes interleaved on the unit circle (alternating A, B, A, B at 90° apart).
    Within-class vectors are opposite each other (cos_dist=2), while between-class
    vectors are orthogonal (cos_dist=1), so BSR = 1/2 < 1.
    A ratio < 1 means within-class pairs are actually further apart than between-class —
    geometry does not encode class structure.
    """
    states = _make_states({
        "a1": [1.0, 0.0],
        "b1": [0.0, 1.0],
        "a2": [-1.0, 0.0],
        "b2": [0.0, -1.0],
    })
    labels = {"a1": "A", "b1": "B", "a2": "A", "b2": "B"}
    ratio = basin_separation_ratio(states, labels)
    assert ratio == ratio, "ratio is NaN"
    assert ratio < 1.0, (
        f"Expected BSR < 1 for interleaved classes (within > between), got {ratio:.4f}"
    )


def test_bsr_single_state_returns_nan() -> None:
    states = _make_states({"a": [1.0, 0.0]})
    labels = {"a": "A"}
    assert math.isnan(basin_separation_ratio(states, labels))


def test_bsr_symmetric() -> None:
    """Swapping class assignments should not change the ratio (by symmetry)."""
    states = _make_states({
        "p": [1.0, 0.0],
        "q": [0.9, 0.1],
        "r": [0.0, 1.0],
        "s": [0.1, 0.9],
    })
    labels_ab = {"p": "A", "q": "A", "r": "B", "s": "B"}
    labels_ba = {"p": "B", "q": "B", "r": "A", "s": "A"}
    ratio_ab = basin_separation_ratio(states, labels_ab)
    ratio_ba = basin_separation_ratio(states, labels_ba)
    assert abs(ratio_ab - ratio_ba) < 1e-10, (
        f"BSR should be symmetric under label swap: {ratio_ab:.6f} vs {ratio_ba:.6f}"
    )


# ---------------------------------------------------------------------------
# intrinsic_dimensionality
# ---------------------------------------------------------------------------

def test_intrinsic_dim_rank_one() -> None:
    """
    All states on the same line → PR = 1.
    """
    states = _make_states({
        "a": [1.0, 0.0, 0.0],
        "b": [2.0, 0.0, 0.0],
        "c": [3.0, 0.0, 0.0],
        "d": [4.0, 0.0, 0.0],
    })
    idim = intrinsic_dimensionality(states)
    assert abs(idim - 1.0) < 1e-6, f"Expected PR=1 for rank-1 data, got {idim:.6f}"


def test_intrinsic_dim_isotropic() -> None:
    """
    States sampled from an isotropic Gaussian in d dimensions → PR ≈ d.
    """
    rng = np.random.default_rng(0)
    d = 8
    n = 200
    vecs = rng.normal(size=(n, d))
    states = {str(i): vecs[i] for i in range(n)}
    idim = intrinsic_dimensionality(states)
    # PR should be close to d; allow ±20% tolerance
    assert abs(idim - d) < 0.2 * d, (
        f"Expected PR ≈ {d} for isotropic {d}-d data, got {idim:.2f}"
    )


def test_intrinsic_dim_two_d_subspace() -> None:
    """
    States in a 2-D plane embedded in 4-D → PR ≈ 2.
    """
    rng = np.random.default_rng(1)
    n = 100
    z = rng.normal(size=(n, 2))
    # Embed into 4D via [z0, z1, 0, 0]
    vecs = np.concatenate([z, np.zeros((n, 2))], axis=1)
    states = {str(i): vecs[i] for i in range(n)}
    idim = intrinsic_dimensionality(states)
    assert abs(idim - 2.0) < 0.3, f"Expected PR ≈ 2 for 2-D subspace in 4-D, got {idim:.2f}"


def test_intrinsic_dim_single_state_nan() -> None:
    states = _make_states({"a": [1.0, 0.0]})
    assert math.isnan(intrinsic_dimensionality(states))


def test_intrinsic_dim_two_states() -> None:
    """Two states span at most 1 dimension → PR = 1."""
    states = _make_states({"a": [1.0, 0.0], "b": [2.0, 0.0]})
    idim = intrinsic_dimensionality(states)
    assert abs(idim - 1.0) < 1e-6, f"Expected PR=1 for 2 collinear states, got {idim:.6f}"


def test_intrinsic_dim_positive() -> None:
    """PR must always be >= 1."""
    rng = np.random.default_rng(42)
    for _ in range(10):
        n = rng.integers(3, 20)
        d = rng.integers(2, 12)
        vecs = rng.normal(size=(n, d))
        states = {str(i): vecs[i] for i in range(n)}
        idim = intrinsic_dimensionality(states)
        assert idim >= 1.0 - 1e-9, f"PR must be >= 1, got {idim:.6f}"


# ---------------------------------------------------------------------------
# Probe script smoke test
# ---------------------------------------------------------------------------

def test_probe_harness_smoke(tmp_path, monkeypatch) -> None:
    """
    Run probe_attractors in harness mode with a short training run.
    Checks that it returns a dict with the expected keys and sensible values.
    """
    import sys
    import importlib
    import types

    # Patch TRAIN_EPOCHS to a small value so the test is fast
    import eval_harness
    original_epochs = eval_harness.TRAIN_EPOCHS

    # We'll monkeypatch at the module level used inside the probe
    monkeypatch.setattr("eval_harness.TRAIN_EPOCHS", 100)

    # Import and call the probe directly
    probe_mod_path = str(_ROOT / "scripts" / "probe_attractors.py")
    spec = importlib.util.spec_from_file_location("probe_attractors", probe_mod_path)
    probe_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe_mod)

    import argparse
    args = argparse.Namespace(
        checkpoint=None,
        max_relax_steps=50,
        relax_tol=1e-4,
        save=False,
        seed=0,
        n_docs=5,
        step_tokens=50,
        bpe_model="data/bpe/wikitext2_8k",
    )
    results = probe_mod._run_harness_probe(args)

    assert results["mode"] == "harness"
    assert "forward_states" in results
    assert "converged_attractors" in results
    assert "convergence" in results
    fwd = results["forward_states"]
    assert fwd["n_states"] == 5
    assert fwd["mean_norm"] > 0.0
    # intrinsic_dim must be computable (5 states in 32-D)
    assert fwd["intrinsic_dimensionality"] is not None
    # The toy corpus has 5 sentences each with a unique subject noun, so all
    # class labels are singletons → BSR is NaN (None). That is correct behaviour.
    # A non-None BSR here would indicate a bug in singleton handling.

    conv = results["convergence"]
    assert conv["n_probed"] == 5
    assert conv["mean_steps"] >= 1


# Expose project root for the smoke test
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
