"""Short-corpus safety checks for text dataset windows."""

from __future__ import annotations

from pathlib import Path

from attractor_llm.tokenizer import AttractorTokenizer
from attractor_llm.training import TextDataset


def test_short_corpus_dataset_padding(short_text_path: Path) -> None:
    """TextDataset should remain usable on very short corpora."""
    tokenizer = AttractorTokenizer(use_tiktoken=False)
    dataset = TextDataset(short_text_path, tokenizer=tokenizer, seq_len=8)
    assert len(dataset) >= 1
    x, y = dataset[0]
    assert len(x) == 8
    assert len(y) == 8

