"""Unit tests for integrated Phase 3 controller/adapter flow."""

from __future__ import annotations

import pytest

from attractor_llm.phase3 import (
    Phase3Adapter,
    Phase3Controller,
    Phase3RuntimeState,
    run_offline_simulation,
)


class _DummyOpt:
    def __init__(self, lr: float = 0.01) -> None:
        self.param_groups = [{"lr": lr}]


def test_phase3_adapter_is_strict_no_fallback() -> None:
    adapter = Phase3Adapter()
    runtime = Phase3RuntimeState()
    opt = _DummyOpt()
    with pytest.raises(ValueError):
        adapter.apply(
            {"action": "not_real", "params": {}, "reason": "test", "ttl_steps": 1},
            optimizer=opt,
            runtime=runtime,
        )


def test_phase3_offline_simulation_emits_controls() -> None:
    snapshots = [
        {
            "epoch": 1,
            "step": 1,
            "train_loss": 1.0,
            "val_loss": None,
            "steps_per_sec": 10.0,
            "grad_norm": 1.0,
            "timestamp_s": 0.0,
        },
        {
            "epoch": 1,
            "step": 2,
            "train_loss": 1.2,
            "val_loss": None,
            "steps_per_sec": 10.0,
            "grad_norm": 1.1,
            "timestamp_s": 0.1,
        },
    ]
    controller = Phase3Controller(enabled=True, budget_steps=10, loss_rise_threshold=0.01)
    adapter = Phase3Adapter()
    results = run_offline_simulation(
        snapshots=snapshots,
        base_lr=0.01,
        controller=controller,
        adapter=adapter,
    )
    assert len(results) == 2
    assert results[1].action == "adjust_lr"
    assert results[1].lr < results[0].lr
