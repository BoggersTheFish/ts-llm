"""Experimental training-time advisor (detached, opt-in, safety-first)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from attractor_llm.phase3.config import SelfImproveConfig


@dataclass(slots=True)
class SelfImproveAdvice:
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
        self.step += 1
        self.history.append(float(loss_value))
        rolling = sum(self.history) / max(len(self.history), 1)
        if (not self.config.enabled) or (self.step < self.config.warmup_batches) or len(self.history) < 2:
            return SelfImproveAdvice(active=False, lr_scale=1.0, clip_scale=1.0, rolling_loss=rolling)

        prev = list(self.history)[-2]
        delta = float(loss_value) - float(prev)
        strength = max(self.config.strength, 0.0)
        # Loss rising -> reduce lr and tighten clipping. Loss falling -> slightly relax.
        lr_scale = 1.0 - max(min(delta * strength, 0.2), -0.2)
        clip_scale = 1.0 - max(min(delta * strength * 0.5, 0.15), -0.15)
        lr_scale = max(0.5, min(1.5, lr_scale))
        clip_scale = max(0.7, min(1.3, clip_scale))
        return SelfImproveAdvice(active=True, lr_scale=lr_scale, clip_scale=clip_scale, rolling_loss=rolling)
