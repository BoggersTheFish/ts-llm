"""Minimal Hopfield energy module for benchmarking (not used in default training)."""

from __future__ import annotations

import torch
import torch.nn as nn


class HopfieldEnergyBaseline(nn.Module):
    """Classical Hopfield update on a projected pattern (diagnostic only)."""

    def __init__(self, dim: int, patterns: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.patterns = patterns
        self.W = nn.Parameter(torch.randn(patterns, dim) * 0.02)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """Hopfield energy ~ -sum tanh(Wx)^2 (scalar per batch row)."""
        z = torch.tanh(x @ self.W.T)
        return -(z**2).sum(dim=-1)
