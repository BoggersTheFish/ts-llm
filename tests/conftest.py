"""Pytest fixtures for fast ts-llm unit tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from attractor_llm.tokenizer import AttractorTokenizer


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("RUN_HEAVY_TESTS") == "1":
        return
    skip_h = pytest.mark.skip(reason="heavy test; set RUN_HEAVY_TESTS=1")
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip_h)

    if os.environ.get("RUN_GPU_TESTS") == "1":
        return
    skip_g = pytest.mark.skip(reason="gpu test; set RUN_GPU_TESTS=1")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_g)


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
