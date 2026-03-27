"""Tokenizer consistency checks for fallback and optional BPE modes."""

from __future__ import annotations

from attractor_llm.tokenizer import AttractorTokenizer


def test_word_tokenizer_roundtrip(word_tokenizer: AttractorTokenizer) -> None:
    """Word-list tokenizer should preserve known-token phrase round-trip."""
    text = "the and time"
    ids = word_tokenizer.encode(text)
    assert ids, "Expected non-empty token IDs in word-list mode."
    decoded = word_tokenizer.decode(ids)
    assert decoded == text


def test_tiktoken_mode_respects_vocab_cap() -> None:
    """When BPE is active, encoded IDs must be below configured cap."""
    tokenizer = AttractorTokenizer(encoding_name="gpt2", vocab_cap=32, use_tiktoken=True)
    if not tokenizer.uses_tiktoken:
        return
    ids = tokenizer.encode("hello world")
    assert all(i < tokenizer.get_vocab_size() for i in ids)

