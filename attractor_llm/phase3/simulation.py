"""Offline Phase 3 simulation harness for replaying metric traces.

Note:
    This module applies controller+adapter logic without running training
    updates, enabling deterministic policy iteration in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from attractor_llm.phase3.adapter import Phase3Adapter, Phase3RuntimeState
from attractor_llm.phase3.controller import Phase3Controller
from attractor_llm.phase3.contracts import Phase3MetricsSnapshot


@dataclass(slots=True)
class OfflineStepResult:
    """Single replay-step result for Phase 3 simulation."""

    step: int
    action: str
    reason: str
    message: str
    lr: float
    clip_scale: float


class _OfflineOptimizer:
    """Minimal optimizer-like object used for offline replay only."""

    def __init__(self, lr: float) -> None:
        self.param_groups: list[dict[str, float]] = [{"lr": float(lr)}]


def run_offline_simulation(
    snapshots: list[Phase3MetricsSnapshot],
    *,
    base_lr: float,
    controller: Phase3Controller,
    adapter: Phase3Adapter,
) -> list[OfflineStepResult]:
    """Replay metrics snapshots through the Phase 3 controller and adapter.

    Args:
        snapshots: Ordered list of metrics snapshots.
        base_lr: Initial learning rate used in replay state.
        controller: Controller instance used to emit decisions.
        adapter: Adapter instance used to apply decisions.

    Returns:
        Per-step replay results with action and effective controls.
    """
    opt = _OfflineOptimizer(lr=base_lr)
    runtime = Phase3RuntimeState()
    out: list[OfflineStepResult] = []
    for idx, snapshot in enumerate(snapshots, start=1):
        decision = controller.decide(snapshot)
        apply_result = adapter.apply(decision, optimizer=opt, runtime=runtime)
        out.append(
            OfflineStepResult(
                step=idx,
                action=decision["action"],
                reason=decision["reason"],
                message=apply_result.message,
                lr=float(opt.param_groups[0]["lr"]),
                clip_scale=runtime.clip_scale,
            )
        )
    return out


def snapshots_from_dicts(rows: list[dict[str, Any]]) -> list[Phase3MetricsSnapshot]:
    """Convert loose dict rows into typed snapshots for replay convenience."""
    snapshots: list[Phase3MetricsSnapshot] = []
    for row in rows:
        snapshots.append(
            {
                "epoch": int(row["epoch"]),
                "step": int(row["step"]),
                "train_loss": float(row["train_loss"]),
                "val_loss": None if row.get("val_loss") is None else float(row["val_loss"]),
                "steps_per_sec": float(row.get("steps_per_sec", 0.0)),
                "grad_norm": None if row.get("grad_norm") is None else float(row["grad_norm"]),
                "timestamp_s": float(row.get("timestamp_s", 0.0)),
            }
        )
    return snapshots
