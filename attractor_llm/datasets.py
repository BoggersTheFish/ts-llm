"""Optional datasets — TinyStories downloads only after explicit user confirmation."""

from __future__ import annotations

import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from attractor_llm.tokenizer import AttractorTokenizer

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]


# Hugging Face dataset artifact (Ronen Eldan / TinyStories).
TINYSTORIES_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
)
TINYSTORIES_DATA_DIR = Path("data/tinystories")

# Cache full token id list per (tokenizer key) so train/val splits share one load.
_TINYSTORIES_IDS_CACHE: dict[tuple[Any, ...], list[int]] = {}


def _tokenizer_cache_key(tokenizer: AttractorTokenizer, max_files: int | None) -> tuple[Any, ...]:
    return (tokenizer.encoding_name, int(tokenizer.n_vocab), tokenizer.uses_tiktoken, max_files)


def _confirm_download(confirm_fn: Callable[[], str] | None) -> bool:
    prompt = "Download TinyStories (~1.2 GB compressed, ~2.5 GB extracted)? [y/N] "
    if confirm_fn is not None:
        return confirm_fn().strip().lower() == "y"
    return input(prompt).strip().lower() == "y"


def _download_tiny_stories_archive(tar_path: Path) -> None:
    if requests is None:
        raise ImportError("The 'requests' package is required to download TinyStories. pip install requests")
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading TinyStories...")
    with requests.get(TINYSTORIES_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        f = open(tar_path, "wb")
        try:
            if tqdm is not None and total > 0:
                pbar = tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc="TinyStories",
                )
            else:
                pbar = None
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    if pbar is not None:
                        pbar.update(len(chunk))
            if pbar is not None:
                pbar.close()
        finally:
            f.close()
    if tar_path.stat().st_size < 10 * 1024 * 1024:
        raise RuntimeError(
            f"TinyStories archive is unexpectedly small ({tar_path.stat().st_size} bytes). "
            "The server may have returned an error page instead of the dataset file."
        )
    print(f"Saved archive to {tar_path}")


def _extract_tiny_stories(tar_path: Path, extracted_dir: Path) -> None:
    extracted_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        # Python 3.12+ tarfile filter
        if hasattr(tarfile, "data_filter"):
            tar.extractall(extracted_dir, filter="data")
        else:
            tar.extractall(extracted_dir)
    print("Extracted TinyStories archive.")


def _collect_story_texts(extracted_dir: Path, max_files: int | None) -> list[str]:
    files = sorted(extracted_dir.rglob("*.txt"))
    if max_files is not None:
        files = files[: max(0, max_files)]
    texts: list[str] = []
    iterator = files
    if tqdm is not None:
        iterator = tqdm(files, desc="Loading stories")
    for f in iterator:
        texts.append(f.read_text(encoding="utf-8", errors="replace"))
    return texts


def load_tinystories_token_ids(
    tokenizer: AttractorTokenizer,
    *,
    data_dir: Path | None = None,
    max_files: int | None = None,
    confirm_fn: Callable[[], str] | None = None,
) -> list[int]:
    """
    Ensure TinyStories is present, return the full concatenated token id sequence.

    Prompts for download confirmation only when data is missing.
    """
    key = _tokenizer_cache_key(tokenizer, max_files)
    if key in _TINYSTORIES_IDS_CACHE:
        return _TINYSTORIES_IDS_CACHE[key]

    base = data_dir or TINYSTORIES_DATA_DIR
    base.mkdir(parents=True, exist_ok=True)
    tar_path = base / "TinyStories_all_data.tar.gz"
    extracted_dir = base / "extracted"

    txt_files = list(extracted_dir.rglob("*.txt")) if extracted_dir.exists() else []
    if not txt_files:
        if not tar_path.is_file():
            if not _confirm_download(confirm_fn):
                raise SystemExit("Download cancelled by user.")
            _download_tiny_stories_archive(tar_path)
        _extract_tiny_stories(tar_path, extracted_dir)
        txt_files = list(extracted_dir.rglob("*.txt"))

    if not txt_files:
        raise RuntimeError(f"No .txt files found under {extracted_dir} after extraction.")

    texts = _collect_story_texts(extracted_dir, max_files)
    full_text = " ".join(texts)
    ids = tokenizer.encode(full_text)
    _TINYSTORIES_IDS_CACHE[key] = ids
    print(f"✅ TinyStories tokenized: {len(ids)} tokens.")
    return ids


class TinyStoriesDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    TinyStories — downloads only when data is missing, after interactive confirmation.

    Train/validation are disjoint slices of the token stream (same scheme as :class:`TextDataset`).
    """

    def __init__(
        self,
        split: str = "train",
        val_split: float = 0.1,
        tokenizer: AttractorTokenizer | None = None,
        seq_len: int = 128,
        max_files: int | None = None,
        data_dir: Path | None = None,
        confirm_fn: Callable[[], str] | None = None,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        if not 0.0 <= val_split < 1.0:
            raise ValueError("val_split must be in [0, 1)")

        self.tokenizer = tokenizer or AttractorTokenizer()
        self.seq_len = seq_len
        self.split = split

        full_ids = load_tinystories_token_ids(
            self.tokenizer,
            data_dir=data_dir,
            max_files=max_files,
            confirm_fn=confirm_fn,
        )
        split_idx = int(len(full_ids) * (1.0 - val_split))
        self.ids = full_ids[:split_idx] if split == "train" else full_ids[split_idx:]

        if len(self.ids) < self.seq_len + 1:
            pad = self.seq_len + 1 - len(self.ids)
            self.ids = self.ids + [0] * pad

    def __len__(self) -> int:
        return max(0, len(self.ids) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        inp = self.ids[idx : idx + self.seq_len]
        tgt = self.ids[idx + 1 : idx + self.seq_len + 1]
        return (
            torch.tensor(inp, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
        )
