"""Hierarchical multi-timescale controller stub (Phase 3 placeholder).

Constraint-graph-driven basin steering will plug in here; default is a no-op policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class ControllerState:
    """Mutable runtime state for future steering hooks."""

    step: int = 0


class Phase3ControllerStub:
    """No-op controller: returns identity adjustments."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.state = ControllerState()

    def on_batch_end(self, metrics: Mapping[str, Any]) -> dict[str, Any]:
        """Inspect metrics; return optional LR/clip multipliers (all 1.0)."""
        self.state.step += 1
        return {"lr_scale": 1.0, "clip_scale": 1.0}

    def steer(
        self,
        state_vector: Any,
        constraint_signals: Mapping[str, float] | None = None,
    ) -> Any:
        """Future: bias dynamics using constraint graph activations."""
        _ = constraint_signals
        return state_vector
