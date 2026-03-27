"""Tests for ``ts_attractor.numpy_demo`` (fast, no GPU)."""

from __future__ import annotations

import numpy as np
import pytest

from ts_attractor import numpy_demo as nd


def test_text_to_signal_unit_norm() -> None:
    v = nd.text_to_signal("hello", 64)
    assert v.shape == (64,)
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-6


def test_step_state_shape() -> None:
    rng = np.random.default_rng(0)
    diff = nd.make_diffusion_matrix(16, rng)
    s = np.zeros(16)
    sig = nd.text_to_signal("x", 16)
    s2 = nd.step_state(s, diff, sig, 0.05)
    assert s2.shape == (16,)


def test_stabilize_norm_band() -> None:
    v = np.ones(8) * 10.0
    out = nd.stabilize_norm(v, 0.5, 3.0, value_clip=5.0)
    assert float(np.linalg.norm(out)) <= 3.0 + 1e-6


def test_proto_logits() -> None:
    s = np.zeros(4)
    att = np.eye(4)[:3]
    logits = nd.proto_logits(s, att, tau=1.0)
    assert logits.shape == (3,)
    assert logits[0] == pytest.approx(-1.0)  # distance to e1 is 1


def test_integrate_finite() -> None:
    rng = np.random.default_rng(1)
    diff = nd.make_diffusion_matrix(32, rng)
    sig = nd.text_to_signal("y", 32)
    s = nd.integrate(diff, sig, dim=32, num_steps=20)
    assert np.all(np.isfinite(s))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_toy_cycle_story_deterministic(seed: int) -> None:
    a = nd.toy_cycle_story(seed=seed)
    b = nd.toy_cycle_story(seed=seed)
    assert a == b


def test_save_load_roundtrip(tmp_path) -> None:
    p = tmp_path / "t.npz"
    nd.save_toy_checkpoint(str(p), dim=24, seed=7)
    d = nd.load_toy_checkpoint(str(p))
    assert int(d["dim"]) == 24
    assert d["final_state"].shape == (24,)
