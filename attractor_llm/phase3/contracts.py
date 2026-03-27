"""Typed contracts for integrated Phase 3 decisions and metric snapshots.

Note:
    Contracts stay additive and explicit so Phase 3 can guide optimization
    without changing attractor-state evolution equations.
"""

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


class Phase3DecisionParams(TypedDict, total=False):
    """Optional payload fields carried by `Phase3Decision`."""

    lr_scale: float
    clip_scale: float
    enabled: bool


class Phase3Decision(TypedDict):
    action: Literal["noop", "adjust_lr", "adjust_clip", "set_constraints"]
    params: Phase3DecisionParams
    reason: str
    ttl_steps: int

