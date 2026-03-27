"""Tests for ``ts_attractor.training_loop``."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ts_attractor.training_loop import save_checkpoint, text_dataloader


def test_text_dataloader_batches() -> None:
    dl = text_dataloader(["a", "b", "c", "d"], batch_size=2)
    batches = list(dl)
    assert len(batches) == 2


def test_save_checkpoint_roundtrip(tmp_path: Path) -> None:
    m = nn.Linear(4, 2)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    p = tmp_path / "ck.pt"
    save_checkpoint(p, m, opt, meta={"x": 1})
    try:
        z = torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        z = torch.load(p, map_location="cpu")
    assert "model" in z and "optimizer" in z
    assert z["meta"]["x"] == 1


def test_train_epoch_amp_signature() -> None:
    from ts_attractor.training_loop import train_epoch_amp

    assert callable(train_epoch_amp)
