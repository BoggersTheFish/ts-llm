"""Training utilities – fully fixed for TinyStories + grad clipping."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from pathlib import Path
from typing import Tuple, List
from .torch_model import TorchAttractorLanguageModel
from .tokenizer import AttractorTokenizer


class TextDataset(Dataset):
    def __init__(self, text_file: str | Path | None = None, tokenizer: AttractorTokenizer | None = None, seq_len: int = 16):
        self.tokenizer = tokenizer or AttractorTokenizer()
        self.seq_len = seq_len
        if text_file and Path(text_file).exists():
            text = Path(text_file).read_text(encoding="utf-8")
            self.ids = self.tokenizer.encode(text)
        else:
            self.ids = list(range(5)) * 1000
        self.ids = self.ids[: len(self.ids) - self.seq_len]

    def __len__(self):
        return len(self.ids) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        x = self.ids[idx : idx + self.seq_len]
        y = self.ids[idx + 1 : idx + self.seq_len + 1]
        return x, y


def save_checkpoint(model: TorchAttractorLanguageModel, path: str | Path, optimizer: optim.Optimizer | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "vocab": model.vocab,
        "state_dim": model.state_dim,
        "optimizer_state": optimizer.state_dict() if optimizer else None,
    }
    torch.save(state, path)
    print(f"✓ Checkpoint saved: {path}")


def load_checkpoint(checkpoint_path: str | Path, device: torch.device = torch.device("cpu")):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = ckpt.get("config") or {}
    state_dim = int(ckpt.get("state_dim", cfg.get("state_dim", 512)))
    vocab = ckpt.get("vocab", cfg.get("vocab"))
    model = TorchAttractorLanguageModel(state_dim=state_dim, vocab=vocab)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.to(device)
    return model, {"optimizer_state": ckpt.get("optimizer_state")}


def train_epoch(
    model: TorchAttractorLanguageModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    max_grad_norm: float | None = 1.0,
):
    """Train one epoch; supports list/tensor batches and optional grad clipping."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids, target_ids = batch

        optimizer.zero_grad()
        loss = model.training_step(input_ids, target_ids)
        loss.backward()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
