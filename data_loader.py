"""
Data loading and document chunking for AttractorLM.

WikiText-2 is loaded via HuggingFace datasets.  Documents are separated at
article-title boundaries (lines starting with '= ... =').  Each document is
encoded in full and yielded as a single long tensor; forward_chunked() handles
the BPTT windowing internally, which keeps h_slow continuous across chunks.

Typical use
-----------
    from tokenizer import BPETokenizer
    from data_loader import WikiText2

    tok = BPETokenizer("data/bpe/wikitext2_8k.model")
    ds = WikiText2("train", tok)
    for doc_ids in ds.iter_documents(shuffle=True):
        loss = model.forward_chunked(doc_ids, chunk_size=64)
        ...
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# WikiText-2
# ---------------------------------------------------------------------------

def _parse_wikitext_documents(lines: list[str], min_chars: int = 100) -> list[str]:
    """
    Group raw WikiText-2 lines into documents.

    Article titles start with '= Text =' (single '=') and mark document
    boundaries.  Blank lines are stripped.  Documents shorter than min_chars
    characters are dropped.
    """
    docs: list[str] = []
    buf: list[str] = []

    def _flush() -> None:
        text = " ".join(buf).strip()
        if len(text) >= min_chars:
            docs.append(text)
        buf.clear()

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        # Article title: '= ... =' or '== ... ==' etc.
        if line.startswith("=") and line.endswith("="):
            _flush()
        else:
            buf.append(line)

    _flush()
    return docs


class WikiText2:
    """
    WikiText-2 dataset loader.

    Args:
        split:      "train", "validation", or "test"
        tokenizer:  BPETokenizer instance (must already be trained)
        min_chars:  Minimum document length in characters; shorter docs are skipped

    The first call to iter_documents() or len() triggers encoding of all
    documents (one-time cost, cached for the lifetime of this object).
    """

    def __init__(
        self,
        split: str,
        tokenizer,
        min_chars: int = 100,
    ) -> None:
        self.split = split
        self.tokenizer = tokenizer
        self.min_chars = min_chars
        self._docs: list[str] | None = None       # raw text
        self._encoded: list[list[int]] | None = None  # token ids

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_raw(self) -> list[str]:
        from datasets import load_dataset  # type: ignore[import]
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=self.split)
        lines = [item["text"] for item in dataset]
        return _parse_wikitext_documents(lines, min_chars=self.min_chars)

    def _ensure_encoded(self) -> None:
        if self._encoded is not None:
            return
        if self._docs is None:
            self._docs = self._load_raw()
        print(
            f"WikiText2 [{self.split}]: encoding {len(self._docs)} documents … ",
            end="",
            flush=True,
        )
        self._encoded = [self.tokenizer.encode(doc) for doc in self._docs]
        total_tok = sum(len(ids) for ids in self._encoded)
        print(f"done  ({total_tok:,} tokens)")

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def iter_documents(
        self,
        shuffle: bool = False,
        min_tokens: int = 2,
    ) -> Iterator[torch.Tensor]:
        """
        Yield one tensor of token ids per document.

        model.forward_chunked(doc_ids) handles BPTT windowing; h is reset
        to zero at the start of each call (= at each document boundary).

        Args:
            shuffle:    Shuffle document order each call (for training)
            min_tokens: Skip documents shorter than this many tokens
        """
        self._ensure_encoded()
        order = list(range(len(self._encoded)))
        if shuffle:
            random.shuffle(order)
        for idx in order:
            ids = self._encoded[idx]
            if len(ids) >= min_tokens:
                yield torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        self._ensure_encoded()
        return len(self._encoded)

    def total_tokens(self) -> int:
        self._ensure_encoded()
        return sum(len(ids) for ids in self._encoded)

    def all_tokens(self) -> list[int]:
        """Flat list of all token ids (for perplexity computation)."""
        self._ensure_encoded()
        result: list[int] = []
        for ids in self._encoded:
            result.extend(ids)
        return result
