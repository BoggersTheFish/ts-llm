"""Checkpoint round-trip and metadata compatibility tests."""

from __future__ import annotations

from pathlib import Path

import torch

from attractor_llm.tokenizer import AttractorTokenizer
from attractor_llm.torch_model import TorchAttractorLanguageModel
from attractor_llm.training import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip_with_metadata(tmp_path: Path) -> None:
    """Saved checkpoint metadata should survive load round-trip."""
    tokenizer = AttractorTokenizer(use_tiktoken=False)
    model = TorchAttractorLanguageModel(
        state_dim=16,
        tokenizer=tokenizer,
        dynamics_type="full",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "roundtrip.pt"

    save_checkpoint(
        model,
        checkpoint_path,
        optimizer=optimizer,
        metadata={"seed": 123, "config": {"state_dim": 16, "source": "pytest"}},
    )

    loaded_model, extras = load_checkpoint(checkpoint_path, device=torch.device("cpu"))
    metadata = extras.get("metadata")

    assert loaded_model.state_dim == 16
    assert isinstance(metadata, dict)
    assert metadata.get("seed") == 123
    assert isinstance(metadata.get("config"), dict)
    assert "temperature" in loaded_model.state_dict()

