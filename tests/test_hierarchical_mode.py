"""Hierarchical configuration safety tests."""

from __future__ import annotations

import pytest

from attractor_llm.tokenizer import AttractorTokenizer
from attractor_llm.torch_model import TorchAttractorLanguageModel


def test_hierarchical_requires_even_state_dim() -> None:
    """Hierarchy mode should reject odd state dimensions."""
    tokenizer = AttractorTokenizer(use_tiktoken=False)
    with pytest.raises(ValueError, match="state_dim must be even"):
        TorchAttractorLanguageModel(
            state_dim=15,
            tokenizer=tokenizer,
            dynamics_type="multihead",
            hierarchy_levels=2,
            num_heads=4,
        )


def test_hierarchical_requires_even_heads() -> None:
    """Hierarchy mode should reject odd head counts."""
    tokenizer = AttractorTokenizer(use_tiktoken=False)
    with pytest.raises(ValueError, match="num_heads must be even"):
        TorchAttractorLanguageModel(
            state_dim=16,
            tokenizer=tokenizer,
            dynamics_type="multihead",
            hierarchy_levels=2,
            num_heads=3,
        )


def test_hierarchical_constructs_with_even_dims() -> None:
    """Hierarchy mode should initialize when parity constraints are met."""
    tokenizer = AttractorTokenizer(use_tiktoken=False)
    model = TorchAttractorLanguageModel(
        state_dim=16,
        tokenizer=tokenizer,
        dynamics_type="multihead",
        hierarchy_levels=2,
        num_heads=4,
    )
    assert model._hierarchy_levels == 2

