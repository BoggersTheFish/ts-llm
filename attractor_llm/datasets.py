"""Optional TinyStories loader — capped token budget + single load shared by train/val."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset
from tqdm import tqdm

from .tokenizer import AttractorTokenizer

# One full token stream per (tokenizer settings, file cap, token cap). Avoids loading
# the corpus twice when constructing train + val splits.
_IDS_CACHE: dict[tuple[Any, ...], list[int]] = {}


def _tokenizer_key(tokenizer: AttractorTokenizer) -> tuple[Any, ...]:
    return (tokenizer.encoding_name, int(tokenizer.n_vocab), tokenizer.uses_tiktoken)


def load_tinystories_token_ids(
    tokenizer: AttractorTokenizer,
    *,
    max_files: int | None = None,
    max_tokens: int | None = None,
    data_dir: Path | None = None,
) -> list[int]:
    """
    Load TinyStories from extracted JSON under ``data/tinystories/extracted``.

    Parameters
    ----------
    max_files :
        Use at most this many JSON shards (by sorted filename).
    max_tokens :
        Stop after this many tokens (after encoding). ``0`` or ``None`` means no limit.
        A finite cap is strongly recommended on CPU — each shard can hold millions of tokens.
    """
    unlimited = max_tokens is None or max_tokens <= 0
    cap = None if unlimited else int(max_tokens)

    cache_key = (_tokenizer_key(tokenizer), max_files, cap)
    if cache_key in _IDS_CACHE:
        return _IDS_CACHE[cache_key]

    base = data_dir or Path("data/tinystories")
    extracted_dir = base / "extracted"

    if not extracted_dir.exists() or not any(extracted_dir.glob("*.json")):
        tar_path = base / "TinyStories_all_data.tar.gz"
        if not tar_path.is_file():
            raise FileNotFoundError(
                f"TinyStories not found: expected {tar_path} or JSON under {extracted_dir}. "
                "Download TinyStories_all_data.tar.gz into data/tinystories/ and extract."
            )
        print("Extracting TinyStories archive (this may take a few minutes)...", flush=True)
        import tarfile

        extracted_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(extracted_dir, filter="data")
            else:
                tar.extractall(extracted_dir)
        print("Extraction complete.", flush=True)

    json_files = sorted(extracted_dir.glob("*.json"), key=lambda p: p.name)
    if not json_files:
        raise RuntimeError(f"No .json files under {extracted_dir}.")

    if max_files is not None and max_files > 0:
        json_files = json_files[: int(max_files)]

    ids: list[int] = []

    def maybe_extend(toks: list[int]) -> bool:
        """Return False if caller should stop loading more text."""
        nonlocal ids
        if cap is None:
            ids.extend(toks)
            return True
        need = cap - len(ids)
        if need <= 0:
            return False
        if len(toks) <= need:
            ids.extend(toks)
            return len(ids) < cap
        ids.extend(toks[:need])
        return False

    for f in tqdm(json_files, desc="Loading stories"):
        if cap is not None and len(ids) >= cap:
            break
        data = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                if cap is not None and len(ids) >= cap:
                    break
                story = item.get("story", "") if isinstance(item, dict) else str(item)
                if story and not maybe_extend(tokenizer.encode(story)):
                    break
        elif isinstance(data, dict):
            story = data.get("story", "")
            if story:
                maybe_extend(tokenizer.encode(story))
        else:
            maybe_extend(tokenizer.encode(str(data)))
        if cap is not None and len(ids) >= cap:
            break

    if cap is not None and len(ids) >= cap:
        print(
            f"TinyStories: using first {len(ids):,} tokens (cap {cap:,}). "
            "Use --tinystories-max-tokens 0 for the full corpus (slow on CPU).",
            flush=True,
        )
    else:
        print(f"TinyStories: loaded {len(ids):,} tokens (unlimited).", flush=True)

    _IDS_CACHE[cache_key] = ids
    return ids


class TinyStoriesDataset(Dataset):
    """Sliding windows over TinyStories token ids (train/val slice of a shared load)."""

    DATA_DIR = Path("data/tinystories")

    def __init__(
        self,
        split: str = "train",
        val_split: float = 0.1,
        tokenizer: AttractorTokenizer | None = None,
        seq_len: int = 64,
        max_files: int | None = None,
        max_tokens: int | None = None,
        max_windows: int | None = None,
        data_dir: Path | None = None,
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
            max_files=max_files,
            max_tokens=max_tokens,
            data_dir=data_dir or self.DATA_DIR,
        )
        split_idx = int(len(full_ids) * (1.0 - val_split))
        self.ids = full_ids[:split_idx] if split == "train" else full_ids[split_idx:]

        # Bound steps per epoch: each window is one batch (batch_size=1). Without this,
        # two JSON shards can still yield tens of millions of windows on CPU.
        if max_windows is not None and int(max_windows) > 0:
            need = int(max_windows) + self.seq_len
            if len(self.ids) > need:
                self.ids = self.ids[:need]
                print(
                    f"TinyStories [{split}]: using first {int(max_windows):,} windows "
                    f"(see --tinystories-max-windows; 0 = no limit).",
                    flush=True,
                )

        if len(self.ids) < self.seq_len + 1:
            pad = self.seq_len + 1 - len(self.ids)
            self.ids = self.ids + [0] * pad

    def __len__(self) -> int:
        return max(0, len(self.ids) - self.seq_len)

    def __getitem__(self, idx: int):
        x = self.ids[idx : idx + self.seq_len]
        y = self.ids[idx + 1 : idx + self.seq_len + 1]
        return x, y
