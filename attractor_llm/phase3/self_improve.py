"""Experimental detached training-time advisor.

Note:
    Advisor consumes scalar losses only and never stores autograd-connected
    tensors, preventing graph-retention side effects.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from attractor_llm.phase3.config import SelfImproveConfig


@dataclass(slots=True)
class SelfImproveAdvice:
    """Structured advisory output for optional optimizer nudges."""

    active: bool
    lr_scale: float
    clip_scale: float
    rolling_loss: float


class SelfImproveAdvisor:
    """
    Stateless-from-autograd advisor that nudges optimizer hyperparameters.

    It only sees scalar loss values (Python floats), never tensors, so it cannot
    introduce graph reuse or backward-through-graph errors.
    """

    def __init__(self, config: SelfImproveConfig) -> None:
        self.config = config
        self.step = 0
        self.history: deque[float] = deque(maxlen=max(config.window, 2))

    def observe(self, loss_value: float) -> SelfImproveAdvice:
        """Observe a scalar loss and emit advisory scaling factors.

        Args:
            loss_value: Latest scalar loss value.

        Returns:
            Advisory payload with optional lr/clip scaling factors.
        """
        self.step += 1
        self.history.append(float(loss_value))
        rolling = sum(self.history) / max(len(self.history), 1)
        if (not self.config.enabled) or (self.step < self.config.warmup_batches) or len(self.history) < 2:
            return SelfImproveAdvice(active=False, lr_scale=1.0, clip_scale=1.0, rolling_loss=rolling)

        history = list(self.history)
        half = max(len(history) // 2, 1)
        older = history[:half]
        newer = history[half:]
        older_mean = sum(older) / max(len(older), 1)
        newer_mean = sum(newer) / max(len(newer), 1)
        delta = float(newer_mean - older_mean)
        strength = max(self.config.strength, 0.0)
        # Loss rising -> reduce lr and tighten clipping. Loss falling -> slightly relax.
        lr_scale = 1.0 - max(min(delta * strength, 0.2), -0.2)
        clip_scale = 1.0 - max(min(delta * strength * 0.5, 0.15), -0.15)
        lr_scale = max(0.5, min(1.5, lr_scale))
        clip_scale = max(0.7, min(1.3, clip_scale))
        return SelfImproveAdvice(active=True, lr_scale=lr_scale, clip_scale=clip_scale, rolling_loss=rolling)
