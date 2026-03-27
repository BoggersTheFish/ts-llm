"""Phase 3 adapter stub for validated, step-boundary decision application."""

from __future__ import annotations

from dataclasses import dataclass

from attractor_llm.phase3.contracts import Phase3Decision


@dataclass(slots=True)
class Phase3ApplyResult:
    applied: bool
    fallback_triggered: bool
    message: str


class Phase3Adapter:
    """
    Adapter boundary between controller decisions and runtime systems.

    This stub does not mutate model state. It validates decisions and returns
    structured results so future integration can remain explicit and safe.
    """

    def __init__(self, safe_fallback: bool = True) -> None:
        self.safe_fallback = bool(safe_fallback)

    def apply(self, decision: Phase3Decision) -> Phase3ApplyResult:
        action = decision.get("action", "noop")
        ttl = int(decision.get("ttl_steps", 0))
        if ttl < 1:
            return Phase3ApplyResult(
                applied=False,
                fallback_triggered=self.safe_fallback,
                message="invalid_ttl_steps",
            )
        if action == "noop":
            return Phase3ApplyResult(applied=True, fallback_triggered=False, message="noop")
        if action in {"adjust_lr", "adjust_clip", "enable_constraint"}:
            # Stub-only path: acknowledge action without mutating runtime.
            return Phase3ApplyResult(applied=True, fallback_triggered=False, message=f"accepted_{action}_stub")
        return Phase3ApplyResult(
            applied=False,
            fallback_triggered=self.safe_fallback,
            message=f"unknown_action_{action}",
        )

