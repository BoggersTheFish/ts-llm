"""Training utilities, checkpoints, and a sliding-window :class:`TextDataset`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from attractor_llm.model import DEFAULT_VOCAB
from attractor_llm.torch_model import TorchAttractorLanguageModel
from attractor_llm.tokenizer import AttractorTokenizer


def _torch_load(path: str | Path, map_location: torch.device | str | None = None) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


class TextDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Sliding windows over token ids from ``text_file`` (UTF-8) or a synthetic stream.

    Uses :class:`~attractor_llm.tokenizer.AttractorTokenizer` for subword or word ids.
    """

    def __init__(
        self,
        text_file: str | Path | None = None,
        *,
        tokenizer: AttractorTokenizer | None = None,
        seq_len: int = 16,
    ) -> None:
        self.tokenizer = tokenizer or AttractorTokenizer(use_tiktoken=False)
        self.seq_len = seq_len

        if text_file is not None and Path(text_file).exists():
            text = Path(text_file).read_text(encoding="utf-8")
            self.ids = self.tokenizer.encode(text)
        else:
            syn = " ".join((["high", "low", "mind", "time", "change"] * 200))
            self.ids = self.tokenizer.encode(syn)

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


def save_checkpoint(
    model: TorchAttractorLanguageModel,
    path: str | Path,
    optimizer: optim.Optimizer | None = None,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save model weights, vocabulary metadata, tokenizer config, and optional optimizer state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = model.config_dict()
    state: dict[str, Any] = {
        "model_state": model.state_dict(),
        "vocab": cfg["vocab"],
        "vocab_size": model.embedder.vocab_size,
        "state_dim": model.state_dim,
        "num_attractor_steps": model.num_attractor_steps,
        "num_converge_steps": model.num_converge_steps,
        "cubic_scale": float(model.dynamics.cubic_scale.detach().cpu().item()),
        "dt": float(model.dynamics.dt),
        "tokenizer_config": cfg.get("tokenizer_config"),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
    }
    if extra:
        state.update(extra)
    torch.save(state, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> tuple[TorchAttractorLanguageModel, dict[str, Any]]:
    """Load a checkpoint produced by :func:`save_checkpoint` or :meth:`TorchAttractorLanguageModel.load`."""
    if device is None:
        device = torch.device("cpu")
    ckpt = _torch_load(checkpoint_path, map_location=device)
    if "model_state" not in ckpt and "state_dict" in ckpt:
        ckpt = {
            "model_state": ckpt["state_dict"],
            "vocab": ckpt.get("vocab", list(DEFAULT_VOCAB)),
            "vocab_size": ckpt.get("vocab_size", len(ckpt.get("vocab", []))),
            "state_dim": ckpt["state_dim"],
            "num_attractor_steps": ckpt.get("num_attractor_steps", 16),
            "num_converge_steps": ckpt.get("num_converge_steps", 12),
            "cubic_scale": ckpt.get("cubic_scale", 0.05),
            "dt": ckpt.get("dt", 0.05),
            "tokenizer_config": ckpt.get("tokenizer_config"),
            "optimizer_state": ckpt.get("optimizer_state"),
        }

    tok_cfg = ckpt.get("tokenizer_config")
    tokenizer: AttractorTokenizer | None = None
    if tok_cfg is not None:
        tokenizer = AttractorTokenizer(
            encoding_name=str(tok_cfg.get("encoding_name", "gpt2")),
            vocab_cap=int(tok_cfg.get("n_vocab", ckpt.get("vocab_size", 8192))),
            use_tiktoken=bool(tok_cfg.get("use_tiktoken", True)),
        )

    vocab_list = list(ckpt.get("vocab") or ([] if tokenizer is not None else list(DEFAULT_VOCAB)))

    model = TorchAttractorLanguageModel(
        state_dim=int(ckpt["state_dim"]),
        vocab=vocab_list,
        tokenizer=tokenizer,
        cubic_scale=float(ckpt.get("cubic_scale", 0.05)),
        dt=float(ckpt.get("dt", 0.05)),
        num_attractor_steps=int(ckpt.get("num_attractor_steps", 16)),
        num_converge_steps=int(ckpt.get("num_converge_steps", 12)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    return model, ckpt


def train_epoch(
    model: TorchAttractorLanguageModel,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: optim.Optimizer,
    device: torch.device,
    *,
    max_grad_norm: float | None = 1.0,
) -> float:
    """
    One epoch: mean loss over batches. Optional global L2 gradient clipping
    (:math:`\\|\\nabla\\|_2 \\leq \\texttt{max_grad_norm}`) before the optimizer step.
    """
    model.train()
    total = 0.0
    n = 0
    for input_ids, target_ids in loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        optimizer.zero_grad(set_to_none=True)
        if input_ids.ndim == 1:
            loss = model.training_step(input_ids, target_ids)
        else:
            bsz = input_ids.shape[0]
            acc = torch.zeros((), device=device)
            for b in range(bsz):
                acc = acc + model.training_step(input_ids[b], target_ids[b])
            loss = acc / max(bsz, 1)
        loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)
