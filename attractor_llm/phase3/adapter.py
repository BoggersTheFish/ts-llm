"""Phase 3 adapter for strict decision application at step boundaries.

Note:
    Adapter actions affect optimizer/runtime control knobs only; attractor
    update equations remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from attractor_llm.phase3.contracts import Phase3Decision


@dataclass(slots=True)
class Phase3ApplyResult:
    """Structured result describing adapter application outcome."""

    applied: bool
    message: str


@dataclass(slots=True)
class Phase3RuntimeState:
    """Mutable runtime control state managed by the Phase 3 adapter."""

    clip_scale: float = 1.0
    constraints_enabled: bool = False


class Phase3Adapter:
    """Boundary between controller decisions and runtime hooks."""

    def apply(
        self,
        decision: Phase3Decision,
        *,
        optimizer: Any,
        runtime: Phase3RuntimeState,
    ) -> Phase3ApplyResult:
        """Validate and apply a Phase 3 decision strictly.

        Args:
            decision: Controller-emitted typed decision payload.
            optimizer: Optimizer instance whose param-group lr may be adjusted.
            runtime: Mutable runtime state for non-lr controls.

        Returns:
            Application result message.

        Raises:
            ValueError: If the decision is invalid or unsupported.
        """
        action = decision.get("action", "noop")
        ttl = int(decision.get("ttl_steps", 0))
        if ttl < 1:
            raise ValueError("invalid_ttl_steps")
        if action == "noop":
            return Phase3ApplyResult(applied=True, message="noop")

        params = decision.get("params", {})
        if action == "adjust_lr":
            scale = float(params.get("lr_scale", 1.0))
            if scale <= 0.0:
                raise ValueError("lr_scale_must_be_positive")
            for group in optimizer.param_groups:
                group["lr"] = float(group["lr"]) * scale
            return Phase3ApplyResult(applied=True, message=f"applied_adjust_lr_{scale:.4f}")

        if action == "adjust_clip":
            scale = float(params.get("clip_scale", 1.0))
            if scale <= 0.0:
                raise ValueError("clip_scale_must_be_positive")
            runtime.clip_scale = max(1e-3, runtime.clip_scale * scale)
            return Phase3ApplyResult(applied=True, message=f"applied_adjust_clip_{scale:.4f}")

        if action == "set_constraints":
            enabled = bool(params.get("enabled", False))
            runtime.constraints_enabled = enabled
            return Phase3ApplyResult(applied=True, message=f"applied_set_constraints_{enabled}")

        raise ValueError(f"unknown_action_{action}")

