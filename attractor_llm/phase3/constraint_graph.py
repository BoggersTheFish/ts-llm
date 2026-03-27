"""Deterministic optional constraint graph for generation-time logit shaping.

Note:
    This module applies additive logit shaping only when enabled and does not
    modify attractor-state dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from attractor_llm.phase3.config import ConstraintConfig


@dataclass(slots=True)
class ConstraintMetrics:
    """Counters for deterministic constraint adjustments."""

    adjustments: int = 0
    total_penalty: float = 0.0


class DeterministicConstraintGraph:
    """
    Minimal deterministic constraint graph.

    Current rule (safe default):
    - If the most recent token has already repeated `max_repeat` times, penalize
      selecting that same token again by subtracting `repeat_penalty` from its logit.
    """

    def __init__(self, config: ConstraintConfig) -> None:
        self.config = config
        self.metrics = ConstraintMetrics()

    def reset_metrics(self) -> None:
        """Reset per-run adjustment counters."""
        self.metrics = ConstraintMetrics()

    def adjust_logits(self, logits: torch.Tensor, history_ids: torch.Tensor) -> torch.Tensor:
        """Apply deterministic repeat-penalty shaping to logits.

        Args:
            logits: Candidate logits tensor.
            history_ids: Generated token history.

        Returns:
            Logits tensor (either unchanged or adjusted copy).
        """
        if not self.config.enabled:
            return logits
        if history_ids.numel() == 0:
            return logits

        last = int(history_ids[-1].item())
        run = 0
        for i in range(int(history_ids.numel()) - 1, -1, -1):
            if int(history_ids[i].item()) == last:
                run += 1
            else:
                break
        if run < self.config.max_repeat:
            return logits

        out = logits.clone()
        out[..., last] = out[..., last] - float(self.config.repeat_penalty)
        self.metrics.adjustments += 1
        self.metrics.total_penalty += float(self.config.repeat_penalty)
        return out
