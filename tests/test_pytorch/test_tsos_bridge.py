"""TS-OS bridge stubs."""

from __future__ import annotations

import torch

from ts_attractor.tsos_bridge import WaveCycleRunner, attach_attractor_state


def test_runner_records() -> None:
    r = WaveCycleRunner()
    attach_attractor_state(r, "a", torch.zeros(3))
    assert r.last() is not None
    assert r.last().node_id == "attractor::a"
