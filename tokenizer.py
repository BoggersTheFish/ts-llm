"""
BPE tokenizer wrapper around SentencePiece for AttractorLM.

Usage
-----
Training (one-time):
    from tokenizer import train_bpe
    train_bpe(texts, model_prefix="data/bpe/wikitext2_8k", vocab_size=8192)

Inference:
    from tokenizer import BPETokenizer
    tok = BPETokenizer("data/bpe/wikitext2_8k.model")
    ids = tok.encode("The cat chases the dog")
    text = tok.decode(ids)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


class BPETokenizer:
    """Thin wrapper around a trained SentencePiece BPE model."""

    def __init__(self, model_path: str) -> None:
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(model_path))

    def encode(self, text: str) -> list[int]:
        return self.sp.Encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self.sp.Decode(ids)

    def vocab_size(self) -> int:
        return self.sp.GetPieceSize()

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()

    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()

    @property
    def unk_id(self) -> int:
        return self.sp.unk_id()

    def piece(self, idx: int) -> str:
        return self.sp.IdToPiece(idx)


def train_bpe(
    texts: list[str],
    model_prefix: str,
    vocab_size: int = 8192,
) -> None:
    """
    Train a SentencePiece BPE model on a list of raw text strings.

    Writes <model_prefix>.model and <model_prefix>.vocab to disk.
    The parent directory is created if it does not exist.

    Args:
        texts:        List of raw text strings (one document per entry).
        model_prefix: Path prefix for output files (without extension).
        vocab_size:   BPE vocabulary size (default 8192 for WikiText-2).
    """
    import sentencepiece as spm

    Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)

    # Write all texts to a temporary file (SentencePiece needs a file path)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as fh:
        for text in texts:
            line = text.strip()
            if line:
                fh.write(line + "\n")
        tmp_path = fh.name

    try:
        spm.SentencePieceTrainer.Train(
            input=tmp_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=0.9995,
            model_type="bpe",
            # Standard special token ids
            pad_id=3,
            eos_id=2,
            bos_id=1,
            unk_id=0,
            # Don't split on whitespace — keep natural word boundaries
            add_dummy_prefix=True,
        )
    finally:
        os.unlink(tmp_path)
