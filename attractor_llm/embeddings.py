"""Learnable signal embeddings — maps token indices to unit-norm vectors in :math:`\\mathbb{R}^D`."""

from __future__ import annotations

import torch
import torch.nn as nn

from attractor_llm.model import DEFAULT_VOCAB


class LearnableProtoEmbedder(nn.Module):
    r"""
    Embedding matrix :math:`E \\in \\mathbb{R}^{V \\times D}` followed by layer norm and
    **row-wise** normalization to the unit sphere:

    .. math::

        u_i = \frac{\mathrm{LN}(E_i)}{\|\mathrm{LN}(E_i)\|_2}.

    Parameters
    ----------
    dim :
        State / signal dimension :math:`D`.
    vocab :
        Optional word list (legacy toy mode). If ``vocab_size`` is set, it overrides
        ``len(vocab)`` for the embedding table width.
    vocab_size :
        If given, embedding has exactly ``vocab_size`` rows (subword / capped vocab).
    """

    def __init__(
        self,
        dim: int = 128,
        vocab: list[str] | None = None,
        *,
        vocab_size: int | None = None,
    ) -> None:
        super().__init__()
        if vocab_size is not None:
            self.vocab_size = int(vocab_size)
            self.vocab = list(vocab) if vocab is not None else []
        else:
            self.vocab = vocab or list(DEFAULT_VOCAB)
            self.vocab_size = len(self.vocab)
        self.dim = dim

        self.embedding = nn.Embedding(self.vocab_size, dim)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-12)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        token_ids :
            Shape ``(batch,)`` or ``(batch, k)`` — long indices in ``[0, V)``.

        Returns
        -------
        torch.Tensor
            Unit-norm signals with the same leading shape as ``token_ids`` and last dim ``dim``.
        """
        signals = self.embedding(token_ids)
        signals = self.norm(signals)
        norms = torch.linalg.vector_norm(signals, dim=-1, keepdim=True)
        return signals / norms.clamp(min=1e-8)

    def get_signal(self, text: str) -> torch.Tensor:
        """Map a single **word** from :attr:`vocab` to a signal (toy / debug)."""
        if not self.vocab:
            raise ValueError("get_signal requires a non-empty vocab list")
        try:
            idx = self.vocab.index(text)
        except ValueError:
            idx = 0
        dev = self.embedding.weight.device
        return self.forward(torch.tensor([idx], dtype=torch.long, device=dev)).squeeze(0)

    def get_all_signals(self) -> torch.Tensor:
        """All rows as ``(V, D)`` — used to build proto-attractors."""
        dev = self.embedding.weight.device
        ids = torch.arange(self.vocab_size, dtype=torch.long, device=dev)
        return self.forward(ids)
