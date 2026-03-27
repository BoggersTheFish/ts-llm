"""Hopfield baseline."""

from __future__ import annotations

import torch

from ts_attractor.baselines.hopfield import HopfieldEnergyBaseline


def test_hopfield_energy_shape() -> None:
    h = HopfieldEnergyBaseline(16, patterns=4)
    x = torch.randn(2, 16)
    e = h.energy(x)
    assert e.shape == (2,)
