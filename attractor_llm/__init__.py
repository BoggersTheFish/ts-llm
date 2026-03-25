"""Nonlinear attractor dynamics: NumPy (legacy) and optional PyTorch (trainable) core."""

from __future__ import annotations

from attractor_llm.core import (
    center,
    converge,
    euclidean_distance,
    linear_diffusion,
    make_diffusion_matrix,
    step_state,
    text_to_signal,
)
from attractor_llm.model import AttractorLanguageModel, GenerationConfig, GenerationResult

__all__ = [
    "text_to_signal",
    "make_diffusion_matrix",
    "linear_diffusion",
    "center",
    "step_state",
    "converge",
    "euclidean_distance",
    "AttractorLanguageModel",
    "GenerationConfig",
    "GenerationResult",
]

try:
    from attractor_llm.embeddings import LearnableProtoEmbedder
    from attractor_llm.torch_core import (
        AttractorDynamics,
        MultiHeadDynamics,
        converge_adaptive,
        converge_fixed_steps,
        text_to_signal as torch_text_to_signal,
    )
    from attractor_llm.torch_model import TorchAttractorLanguageModel, synthetic_training_demo
    from attractor_llm.tokenizer import AttractorTokenizer
    from attractor_llm.training import (
        TextDataset,
        load_checkpoint,
        save_checkpoint,
        train_epoch,
    )
except ImportError:  # pragma: no cover - optional until torch is installed
    pass
else:
    __all__ += [
        "AttractorDynamics",
        "MultiHeadDynamics",
        "torch_text_to_signal",
        "converge_adaptive",
        "converge_fixed_steps",
        "LearnableProtoEmbedder",
        "TorchAttractorLanguageModel",
        "synthetic_training_demo",
        "AttractorTokenizer",
        "TextDataset",
        "save_checkpoint",
        "load_checkpoint",
        "train_epoch",
    ]
