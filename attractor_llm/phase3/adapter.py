"""Phase 3 adapter stub for validated decision application.

Note:
    Current implementation validates decisions but does not mutate runtime
    attractor behavior. This preserves baseline training/generation semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

from attractor_llm.phase3.contracts import Phase3Decision


@dataclass(slots=True)
class Phase3ApplyResult:
    """Structured result describing adapter application outcome."""

    applied: bool
    fallback_triggered: bool
    message: str


class Phase3Adapter:
    """Boundary between controller decisions and runtime hooks.

    Args:
        safe_fallback: Whether unknown/invalid decisions should trigger fallback.

    Note:
        This stub validates only; it intentionally performs no model mutation.
    """

    def __init__(self, safe_fallback: bool = True) -> None:
        self.safe_fallback = bool(safe_fallback)

    def apply(self, decision: Phase3Decision) -> Phase3ApplyResult:
        """Validate and accept/reject a Phase 3 decision.

        Args:
            decision: Controller-emitted typed decision payload.

        Returns:
            Application result including fallback state and message.
        """
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

