"""``ts_attractor`` — modular attractor substrate (NumPy demo + PyTorch wrappers)."""

from __future__ import annotations

__version__ = "0.1.0"

from ts_attractor import numpy_demo
from ts_attractor.controller import Phase3ControllerStub
from ts_attractor.dynamics import AttractorDynamics, MultiHeadDynamics
from ts_attractor.proto_attractors import LearnableProtoEmbedder

__all__ = [
    "__version__",
    "numpy_demo",
    "AttractorDynamics",
    "MultiHeadDynamics",
    "LearnableProtoEmbedder",
    "Phase3ControllerStub",
]
