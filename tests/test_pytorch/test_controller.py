"""Controller stub tests."""

from __future__ import annotations

from ts_attractor.controller import Phase3ControllerStub


def test_controller_noop() -> None:
    c = Phase3ControllerStub()
    adj = c.on_batch_end({"train_loss": 1.0})
    assert adj["lr_scale"] == 1.0
    x = object()
    assert c.steer(x) is x
