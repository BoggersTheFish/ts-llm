"""Pytest fixtures for fast ts-llm unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from attractor_llm.tokenizer import AttractorTokenizer


@pytest.fixture
def word_tokenizer() -> AttractorTokenizer:
    """Provide deterministic word-list tokenizer fixture."""
    return AttractorTokenizer(use_tiktoken=False)


@pytest.fixture
def short_text_path(tmp_path: Path) -> Path:
    """Provide short-corpus text fixture path."""
    p = tmp_path / "short.txt"
    p.write_text("the", encoding="utf-8")
    return p

