"""Phase 3 controller stub (opt-in, no active behavior by default)."""

from __future__ import annotations

from dataclasses import dataclass

from attractor_llm.phase3.contracts import Phase3Decision, Phase3MetricsSnapshot


@dataclass(slots=True)
class Phase3Controller:
    """
    Isolated policy layer for generating bounded control decisions.

    This default controller is intentionally conservative and returns `noop`.
    It is a safe starting point for future policy experiments.
    """

    enabled: bool = False
    budget_steps: int = 0
    consumed_steps: int = 0

    def _budget_exhausted(self) -> bool:
        return self.budget_steps > 0 and self.consumed_steps >= self.budget_steps

    def decide(self, snapshot: Phase3MetricsSnapshot) -> Phase3Decision:
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

