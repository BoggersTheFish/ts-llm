"""Proto embedder wiring."""

from __future__ import annotations

import pytest
import torch

from ts_attractor.proto_attractors import LearnableProtoEmbedder


@pytest.mark.parametrize("vocab, dim", [(16, 32), (32, 64)])
def test_embedder_signals_shape(vocab: int, dim: int) -> None:
    e = LearnableProtoEmbedder(dim=dim, vocab_size=vocab)
    s = e.get_all_signals()
    assert s.shape == (vocab, dim)
