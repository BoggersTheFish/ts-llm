"""Training utilities – fully fixed for TinyStories + grad clipping."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from pathlib import Path
from typing import Tuple, List
from time import perf_counter
from .torch_model import TorchAttractorLanguageModel
from .tokenizer import AttractorTokenizer


class TextDataset(Dataset):
    def __init__(
        self,
        text_file: str | Path | None = None,
        tokenizer: AttractorTokenizer | None = None,
        seq_len: int = 16,
        split: str = "train",
        val_split: float = 0.0,
    ):
        self.tokenizer = tokenizer or AttractorTokenizer()
        self.seq_len = seq_len
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        if not 0.0 <= val_split < 1.0:
            raise ValueError("val_split must be in [0, 1)")
        if text_file and Path(text_file).exists():
            text = Path(text_file).read_text(encoding="utf-8")
            self.ids = self.tokenizer.encode(text)
        else:
            self.ids = list(range(5)) * 1000
        if val_split > 0.0:
            split_idx = int(len(self.ids) * (1.0 - val_split))
            self.ids = self.ids[:split_idx] if split == "train" else self.ids[split_idx:]
        # Keep full token stream; sliding windows are formed in __getitem__.
        if len(self.ids) < self.seq_len + 1:
            self.ids = self.ids + [0] * (self.seq_len + 1 - len(self.ids))

    def __len__(self):
        return max(0, len(self.ids) - self.seq_len)

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
        "config": model.config_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
    }
    torch.save(state, path)
    print(f"✓ Checkpoint saved: {path}")


def load_checkpoint(checkpoint_path: str | Path, device: torch.device = torch.device("cpu")):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = ckpt.get("config") or {}
    state_dim = int(ckpt.get("state_dim", cfg.get("state_dim", 512)))
    vocab = ckpt.get("vocab", cfg.get("vocab"))

    tokenizer = None
    tok_cfg = cfg.get("tokenizer_config")
    if isinstance(tok_cfg, dict):
        tokenizer = AttractorTokenizer(
            encoding_name=str(tok_cfg.get("encoding_name", "gpt2")),
            vocab_cap=int(tok_cfg.get("n_vocab", 8192)),
            use_tiktoken=bool(tok_cfg.get("use_tiktoken", True)),
        )
    elif not vocab:
        # Backward-compatible fallback for checkpoints that predate config serialization.
        tokenizer = AttractorTokenizer(use_tiktoken=False)

    model = TorchAttractorLanguageModel(
        state_dim=state_dim,
        vocab=vocab,
        tokenizer=tokenizer,
        dynamics_type=cfg.get("dynamics_type"),
        cubic_scale=float(cfg.get("cubic_scale", 0.05)),
        dt=float(cfg.get("dt", 0.05)),
        num_heads=int(cfg.get("num_heads", 4)),
        rank=int(cfg.get("rank", 64)) if cfg.get("rank") is not None else 64,
        coupling=float(cfg.get("coupling")) if cfg.get("coupling") is not None else 0.01,
        ode_solver=str(cfg.get("ode_solver", "euler")),
        adaptive_ode=bool(cfg.get("adaptive_ode", False)),
        ode_atol=float(cfg.get("ode_atol", 1e-4)),
        ode_rtol=float(cfg.get("ode_rtol", 1e-4)),
        num_attractor_steps=int(cfg.get("num_attractor_steps", 16)),
        num_converge_steps=int(cfg.get("num_converge_steps", 12)),
        hierarchy_levels=int(cfg.get("hierarchy_levels", 1)),
        timescale_ratio=float(cfg.get("timescale_ratio", 4.0)),
        phrase_vocab_size=int(cfg.get("phrase_vocab_size", 8192)),
        phrase_span=int(cfg.get("phrase_span", 4)),
        phrase_attractors=bool(cfg.get("phrase_attractors", True)),
    )
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.to(device)
    return model, {"optimizer_state": ckpt.get("optimizer_state")}


def train_epoch(
    model: TorchAttractorLanguageModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    max_grad_norm: float | None = 1.0,
    *,
    progress: bool = False,
    epoch: int = 0,
    total_epochs: int = 1,
    log_every_batches: int = 0,
):
    """Train one epoch; supports list/tensor batches and optional grad clipping."""
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[misc, assignment]

    model.train()
    total_loss = 0.0
    iterator = loader
    if progress and tqdm is not None and len(loader) > 0:
        iterator = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{total_epochs}",
            unit="batch",
            leave=True,
        )
    elif progress and tqdm is None:
        print(
            f"[train] epoch {epoch + 1}/{total_epochs}: {len(loader)} batches (install tqdm for a progress bar)",
            flush=True,
        )

    n_batches = 0
    epoch_start = perf_counter()
    window_start = epoch_start
    window_batches = 0
    for batch in iterator:
        input_ids, target_ids = batch

        optimizer.zero_grad()
        loss = model.training_step(input_ids, target_ids)
        loss.backward()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        window_batches += 1

        if log_every_batches > 0 and n_batches % log_every_batches == 0:
            now = perf_counter()
            elapsed = max(now - window_start, 1e-9)
            overall = max(now - epoch_start, 1e-9)
            bps = window_batches / elapsed
            avg_bps = n_batches / overall
            msg = (
                f"[train] epoch {epoch + 1}/{total_epochs} "
                f"batch {n_batches}/{len(loader)} "
                f"loss={loss.item():.4f} bps={bps:.2f} avg_bps={avg_bps:.2f}"
            )
            if progress and tqdm is not None:
                tqdm.write(msg)
            else:
                print(msg, flush=True)
            window_start = now
            window_batches = 0
    return total_loss / max(n_batches, 1)
