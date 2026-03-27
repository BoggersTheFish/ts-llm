"""Hierarchical proto-concepts: multi-timescale signals and split-head dynamics."""

from __future__ import annotations

import torch
import torch.nn as nn

from attractor_llm.embeddings import LearnableProtoEmbedder
from attractor_llm.torch_core import MultiHeadDynamics, _apply_target_norm, _stabilize_state


def rolling_phrase_id(
    input_ids: torch.Tensor,
    t: int,
    phrase_span: int,
    phrase_vocab_size: int,
) -> int:
    """Deterministic phrase index from the last ``phrase_span`` tokens (inclusive)."""
    start = max(0, t - phrase_span + 1)
    window = input_ids[start : t + 1].to(torch.int64)
    if window.numel() == 0:
        return 0
    w = torch.arange(1, window.numel() + 1, device=window.device, dtype=torch.int64)
    return int((window * w).sum().item()) % phrase_vocab_size


class HierarchicalProtoEmbedder(nn.Module):
    """
    Level 0 (fast): token-level unit-norm signals in ``D_fast``.
    Level 1 (slow): phrase-level unit-norm signals in ``D_slow``.

    Combined injection is ``concat(u_fast, u_slow) ∈ R^{D_fast + D_slow}`` (``state_dim``).
    """

    def __init__(
        self,
        state_dim: int,
        *,
        tokenizer_vocab_size: int,
        phrase_vocab_size: int = 8192,
        vocab: list[str] | None = None,
    ) -> None:
        super().__init__()
        if state_dim % 2 != 0:
            raise ValueError("state_dim must be even for hierarchical (fast/slow) split")
        self.state_dim = state_dim
        self.d_fast = state_dim // 2
        self.d_slow = state_dim // 2
        self.phrase_vocab_size = int(phrase_vocab_size)

        self.embed_fast = LearnableProtoEmbedder(
            dim=self.d_fast,
            vocab=vocab,
            vocab_size=tokenizer_vocab_size,
        )
        self.embed_slow = LearnableProtoEmbedder(
            dim=self.d_slow,
            vocab=[],
            vocab_size=self.phrase_vocab_size,
        )

    @property
    def vocab_size(self) -> int:
        """Token vocabulary width (same as :attr:`embed_fast.vocab_size`)."""
        return self.embed_fast.vocab_size

    def signals_from_token_and_phrase(self, token_id: int, phrase_id: int, *, device: torch.device) -> torch.Tensor:
        """``(state_dim,)`` combined signal for one position."""
        tid = torch.tensor([token_id], dtype=torch.long, device=device)
        pid = torch.tensor([phrase_id % self.phrase_vocab_size], dtype=torch.long, device=device)
        uf = self.embed_fast(tid).squeeze(0)
        us = self.embed_slow(pid).squeeze(0)
        return torch.cat([uf, us], dim=-1)

    def get_all_signals(self) -> torch.Tensor:
        """``(V_token, state_dim)`` — slow row ``v % phrase_vocab_size`` paired with token ``v``."""
        fast = self.embed_fast.get_all_signals()
        v = fast.shape[0]
        dev = fast.device
        idx = torch.arange(v, dtype=torch.long, device=dev) % self.phrase_vocab_size
        slow_rows = self.embed_slow(idx)
        return torch.cat([fast, slow_rows], dim=-1)


class MultiTimescaleMultiHeadDynamics(nn.Module):
    """
    Two coupled half-state blocks: **fast** and **slow** :class:`MultiHeadDynamics`,
    each on ``state_dim/2`` with ``num_heads/2`` heads. The slow block uses smaller
    :math:`\\Delta t` and weaker cubic gain (``1/\\texttt{timescale_ratio}``) so it
    evolves more slowly under the same number of discrete steps.
    """

    def __init__(
        self,
        state_dim: int,
        num_heads: int,
        *,
        rank: int = 64,
        cubic_scale: float = 0.05,
        dt: float = 0.05,
        coupling: float = 0.01,
        timescale_ratio: float = 4.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if state_dim % 2 != 0:
            raise ValueError("state_dim must be even")
        if num_heads % 2 != 0:
            raise ValueError("num_heads must be even for multi-timescale dynamics")
        if timescale_ratio <= 0:
            raise ValueError("timescale_ratio must be positive")

        self.state_dim = state_dim
        self.timescale_ratio = float(timescale_ratio)
        d = state_dim // 2
        h = num_heads // 2

        self.fast = MultiHeadDynamics(
            state_dim=d,
            num_heads=h,
            rank=rank,
            cubic_scale=cubic_scale,
            dt=dt,
            coupling=coupling,
            device=device,
            dtype=dtype,
        )
        self.slow = MultiHeadDynamics(
            state_dim=d,
            num_heads=h,
            rank=rank,
            cubic_scale=cubic_scale / timescale_ratio,
            dt=dt / timescale_ratio,
            coupling=coupling,
            device=device,
            dtype=dtype,
        )

    def split_state_signal(
        self,
        state: torch.Tensor,
        signal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        d = self.state_dim // 2
        if state.shape[-1] != self.state_dim or signal.shape[-1] != self.state_dim:
            raise ValueError("state and signal last dim must equal state_dim")
        return state[..., :d], state[..., d:], signal[..., :d], signal[..., d:]

    def drift(self, state: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
        """Stacked drift (same time axis as submodules)."""
        sf, ss, uf, us = self.split_state_signal(state, signal)
        vf = self.fast.drift(sf, uf)
        vs = self.slow.drift(ss, us)
        return torch.cat([vf, vs], dim=-1)

    def unified_drift(self, state: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
        """Single global time field for :mod:`torchdiffeq` (slow channel scaled)."""
        sf, ss, uf, us = self.split_state_signal(state, signal)
        vf = self.fast.drift(sf, uf)
        vs = self.slow.drift(ss, us)
        return torch.cat([vf, vs / self.timescale_ratio], dim=-1)

    def forward(self, state: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
        """One Euler step per block with its own ``dt``."""
        sf, ss, uf, us = self.split_state_signal(state, signal)
        sf2 = self.fast.forward(sf, uf)
        ss2 = self.slow.forward(ss, us)
        return torch.cat([sf2, ss2], dim=-1)

    def converge_fixed(
        self,
        signal: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        *,
        num_steps: int = 20,
        magnitude_floor: float = 1e-3,
        magnitude_ceiling: float | None = 12.0,
        target_norm: float | None = 1.0,
        state_clip_value: float | None = None,
    ) -> torch.Tensor:
        """Fixed-step convergence; same contract as :meth:`MultiHeadDynamics.converge_fixed`."""
        if signal.ndim == 2:
            v, d = signal.shape
            if d != self.state_dim:
                raise ValueError("signal last dim must equal state_dim")
            s = torch.zeros(v, d, device=signal.device, dtype=signal.dtype) if initial_state is None else initial_state.clone()
            for _ in range(num_steps):
                s_next = self.forward(s, signal)
                s = _stabilize_state(
                    s_next,
                    magnitude_floor,
                    magnitude_ceiling,
                    value_clip=state_clip_value,
                )
            return _apply_target_norm(s, d, target_norm)

        dim = signal.shape[0]
        s = torch.zeros(dim, device=signal.device, dtype=signal.dtype) if initial_state is None else initial_state.clone()
        for _ in range(num_steps):
            s_next = self.forward(s, signal)
            s = _stabilize_state(
                s_next,
                magnitude_floor,
                magnitude_ceiling,
                value_clip=state_clip_value,
            )
        return _apply_target_norm(s, dim, target_norm)

    def vector_field(self, state: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
        """Alias of :meth:`drift` for ODE dispatch."""
        return self.drift(state, signal)
