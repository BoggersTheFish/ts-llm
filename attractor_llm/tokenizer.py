"""Tokenizer utilities with tiktoken and deterministic word fallback.

Note:
    Token IDs are capped to a bounded vocabulary width to keep embedding/logit
    tables manageable while preserving deterministic decode behavior.
"""

from __future__ import annotations

from typing import Sequence

import tiktoken

from attractor_llm.model import DEFAULT_VOCAB


class AttractorTokenizer:
    """Tokenizer with optional GPT-style BPE and word-list fallback.

    Args:
        encoding_name: tiktoken encoding name (for example, ``"gpt2"``).
        vocab_cap: Optional maximum exposed token ID width.
        use_tiktoken: Whether to attempt tiktoken mode.

    Note:
        When tiktoken is disabled/unavailable, deterministic word-list mode is used.
    """

    def __init__(
        self,
        *,
        encoding_name: str = "gpt2",
        vocab_cap: int | None = 8192,
        use_tiktoken: bool = True,
    ) -> None:
        self.encoding_name = encoding_name
        self._vocab_cap = vocab_cap
        self._enc: tiktoken.Encoding | None = None
        self._words: list[str] = list(DEFAULT_VOCAB)
        self._word2id: dict[str, int] = {w: i for i, w in enumerate(self._words)}

        if use_tiktoken:
            try:
                enc = tiktoken.get_encoding(encoding_name)
                n_full = enc.n_vocab
                self._enc = enc
                self.n_vocab = int(min(vocab_cap or n_full, n_full))
            except Exception:
                self._enc = None
                self.n_vocab = len(self._words)
        else:
            self.n_vocab = len(self._words)

    @property
    def uses_tiktoken(self) -> bool:
        """True when BPE encoding is active."""
        return self._enc is not None

    def encode(self, text: str) -> list[int]:
        """
        Encode ``text`` to a list of integer ids in ``[0, n_vocab)``.

        In tiktoken mode, ids above ``n_vocab - 1`` are **dropped** (not remapped) so
        decoding stays well-defined.
        """
        if self._enc is not None:
            raw = self._enc.encode(text)
            return [t for t in raw if t < self.n_vocab]
        return [self._word2id[t] for t in text.split() if t in self._word2id]

    def decode(self, ids: Sequence[int]) -> str:
        """
        Decode token ids to text. In tiktoken mode, uses :meth:`tiktoken.Encoding.decode`.
        In word mode, joins known words (unknown indices become ``<unk>``).
        """
        if self._enc is not None:
            safe = [int(i) for i in ids if 0 <= int(i) < self._enc.n_vocab]
            if not safe:
                return ""
            return self._enc.decode(safe)
        parts: list[str] = []
        for i in ids:
            ii = int(i)
            if 0 <= ii < len(self._words):
                parts.append(self._words[ii])
            else:
                parts.append("<unk>")
        return " ".join(parts)

    def get_vocab_size(self) -> int:
        """Alias for :attr:`n_vocab` (embedding / logits width)."""
        return int(self.n_vocab)
