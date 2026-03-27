"""Typed contracts for isolated Phase 3 decisions and metric snapshots."""

from __future__ import annotations

from typing import Literal, TypedDict


class Phase3MetricsSnapshot(TypedDict):
    epoch: int
    step: int
    train_loss: float
    val_loss: float | None
    steps_per_sec: float
    grad_norm: float | None
    timestamp_s: float


class Phase3Decision(TypedDict):
    action: Literal["noop", "adjust_lr", "adjust_clip", "enable_constraint"]
    params: dict[str, float | int | bool]
    reason: str
    ttl_steps: int

