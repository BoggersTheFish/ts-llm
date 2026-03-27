"""Unit tests for integrated Phase 3 controller/adapter flow."""

from __future__ import annotations

import pytest
import torch

from attractor_llm.phase3 import (
    Phase3Adapter,
    Phase3Controller,
    Phase3RuntimeState,
    run_offline_simulation,
    snapshots_from_dicts,
)
from attractor_llm.phase3.config import ConstraintConfig
from attractor_llm.phase3.constraint_graph import DeterministicConstraintGraph


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


def test_phase3_trace_schema_validation_raises() -> None:
    with pytest.raises(ValueError, match="missing required keys"):
        snapshots_from_dicts([{"epoch": 1, "step": 1}])


def test_phase3_constraint_graph_penalizes_repeat() -> None:
    graph = DeterministicConstraintGraph(ConstraintConfig(enabled=True, max_repeat=2, repeat_penalty=1.5))
    logits = torch.tensor([[0.0, 0.1, 0.2]], dtype=torch.float32)
    history = torch.tensor([2, 2], dtype=torch.long)
    adjusted = graph.adjust_logits(logits, history)
    assert float(adjusted[0, 2].item()) == pytest.approx(float(logits[0, 2].item() - 1.5), rel=1e-6)
