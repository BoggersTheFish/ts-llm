"""Training utilities; AMP hooks are stubs until integrated into ``attractor_llm.training``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from attractor_llm.training import train_epoch as _train_epoch


def train_epoch_amp(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    grad_clip: float | None = 1.0,
    use_amp: bool = False,
    scaler: object | None = None,
    checkpoint_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> float:
    """Run one epoch. ``use_amp`` / ``scaler`` are reserved for CUDA AMP (not wired yet).

    ``checkpoint_fn`` is reserved for gradient checkpointing around the forward pass.
    """
    _ = use_amp, scaler, checkpoint_fn
    return float(
        _train_epoch(
            model,  # type: ignore[arg-type]
            loader,
            optimizer,
            device,
            max_grad_norm=grad_clip,
        )
    )


def text_dataloader(
    texts: list[str],
    *,
    batch_size: int = 4,
    collate_fn: Callable[[list[Any]], Any] | None = None,
) -> DataLoader:
    """Minimal arbitrary-text loader (list of strings)."""
    from torch.utils.data import Dataset

    class _Text(Dataset):
        def __init__(self, items: list[str]) -> None:
            self._items = items

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, i: int) -> str:
            return self._items[i]

    return DataLoader(
        _Text(texts),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn if collate_fn is not None else lambda b: b,
    )


def save_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    meta: dict[str, Any] | None = None,
) -> None:
    """Save weights + optional metadata."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"model": model.state_dict()}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if meta:
        payload["meta"] = meta
    torch.save(payload, p)
