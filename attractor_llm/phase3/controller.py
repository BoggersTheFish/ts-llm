"""Phase 3 controller stub with explicit, safe defaults.

Note:
    This controller is currently spec-first scaffolding and returns no-op
    decisions unless explicitly enabled in future integration stages.
"""

from __future__ import annotations

from dataclasses import dataclass

from attractor_llm.phase3.contracts import Phase3Decision, Phase3MetricsSnapshot


@dataclass(slots=True)
class Phase3Controller:
    """Isolated policy layer producing bounded Phase 3 decisions.

    Args:
        enabled: Whether decision generation is active.
        budget_steps: Optional hard cap on decisions produced.
        consumed_steps: Internal counter tracking consumed budget.

    Note:
        Default behavior is intentionally conservative and no-op.
    """

    enabled: bool = False
    budget_steps: int = 0
    consumed_steps: int = 0

    def _budget_exhausted(self) -> bool:
        return self.budget_steps > 0 and self.consumed_steps >= self.budget_steps

    def decide(self, snapshot: Phase3MetricsSnapshot) -> Phase3Decision:
        """Produce the next control decision from current metrics.

        Args:
            snapshot: Immutable metrics snapshot for the current step.

        Returns:
            A typed Phase 3 decision payload.
        """
        _ = snapshot
        if (not self.enabled) or self._budget_exhausted():
            return {
                "action": "noop",
                "params": {},
                "reason": "phase3_disabled_or_budget_exhausted",
                "ttl_steps": 1,
            }
        self.consumed_steps += 1
        return {
            "action": "noop",
            "params": {},
            "reason": "controller_stub_noop",
            "ttl_steps": 1,
        }

