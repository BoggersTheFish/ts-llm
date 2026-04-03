"""
Subword tokenization with :mod:`tiktoken`, with a **word-list fallback** that matches the
legacy :data:`~attractor_llm.model.DEFAULT_VOCAB` toy setup.

When ``tiktoken`` is available, text is encoded with a BPE vocabulary (default **GPT-2**
``gpt2`` encoding). Token ids are folded into ``[0, n)`` where ``n = min(vocab_cap, enc.n_vocab)``
via ``t`` if ``t < n`` else ``t % n`` so the embedding table stays finite **without** collapsing
every OOV-range id onto a single row (which ``clamp`` to ``n-1`` did). Dropping out-of-range
ids also collapsed unrelated strings when ``n`` was small.
"""

from __future__ import annotations

from typing import Sequence

import tiktoken

from attractor_llm.model import DEFAULT_VOCAB


class AttractorTokenizer:
    r"""
    Parameters
    ----------
    encoding_name :
        ``tiktoken`` encoding name (e.g. ``"gpt2"``). Ignored in word fallback mode.
    vocab_cap :
        Maximum number of distinct token ids exposed to the model (embedding rows).
        If ``None``, uses the encoder's full vocabulary size (capped by the encoding).
    use_tiktoken :
        If ``False``, always use the word-list fallback (offline / deterministic).
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

        In tiktoken mode, ids ``t >= n_vocab`` are mapped with ``t % n_vocab`` (ids in range
        are unchanged). This keeps the full BPE length/order while spreading high ids across
        embedding rows; clamping them all to ``n_vocab - 1`` made many unrelated prompts
        identical.
        """
        if self._enc is not None:
            raw = self._enc.encode(text)
            n = self.n_vocab
            if n <= 0:
                return []
            return [t if t < n else (t % n) for t in raw]
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
