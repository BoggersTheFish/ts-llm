"""Phase 3 controller producing bounded optimization control decisions.

Note:
    The controller consumes scalar metrics snapshots only and never touches
    attractor-state tensors directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from attractor_llm.phase3.contracts import Phase3Decision, Phase3MetricsSnapshot


@dataclass(slots=True)
class Phase3Controller:
    """Policy layer producing bounded Phase 3 decisions.

    Args:
        enabled: Whether decision generation is active.
        budget_steps: Optional hard cap on decisions produced.
        loss_rise_threshold: Relative increase threshold that triggers damping.
        loss_drop_threshold: Relative decrease threshold that triggers relaxation.
        consumed_steps: Internal counter tracking consumed budget.
        last_train_loss: Last observed train loss for trend estimation.

    Note:
        Decision outputs are additive optimizer controls and do not mutate model
        architecture or attractor dynamics functions.
    """

    enabled: bool = False
    budget_steps: int = 0
    loss_rise_threshold: float = 0.02
    loss_drop_threshold: float = 0.02
    consumed_steps: int = 0
    last_train_loss: Optional[float] = None

    def _budget_exhausted(self) -> bool:
        return self.budget_steps > 0 and self.consumed_steps >= self.budget_steps

    def decide(self, snapshot: Phase3MetricsSnapshot) -> Phase3Decision:
        """Produce the next control decision from current metrics.

        Args:
            snapshot: Immutable metrics snapshot for the current step.

        Returns:
            A typed Phase 3 decision payload.
        """
        if (not self.enabled) or self._budget_exhausted():
            return {
                "action": "noop",
                "params": {},
                "reason": "phase3_disabled_or_budget_exhausted",
                "ttl_steps": 1,
            }

        self.consumed_steps += 1
        current_loss = float(snapshot["train_loss"])
        if self.last_train_loss is None:
            self.last_train_loss = current_loss
            return {"action": "noop", "params": {}, "reason": "warmup", "ttl_steps": 1}

        baseline = max(abs(self.last_train_loss), 1e-9)
        rel_delta = (current_loss - self.last_train_loss) / baseline
        self.last_train_loss = current_loss

        if rel_delta > self.loss_rise_threshold:
            return {
                "action": "adjust_lr",
                "params": {"lr_scale": 0.9},
                "reason": "loss_rising",
                "ttl_steps": 8,
            }
        if rel_delta < -self.loss_drop_threshold:
            return {
                "action": "adjust_clip",
                "params": {"clip_scale": 1.05},
                "reason": "loss_falling",
                "ttl_steps": 4,
            }
        return {"action": "noop", "params": {}, "reason": "stable_loss", "ttl_steps": 1}

