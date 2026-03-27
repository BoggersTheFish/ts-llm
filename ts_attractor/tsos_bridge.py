"""Bridge between attractor state and TS-OS ``WaveCycleRunner``-style nodes (stub).

If TS-OS is not present in-process, this module exposes stable dataclasses and IDs
so callers can serialize orbit transitions without importing external packages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WaveCycleNode:
    """Single node in a conceptual thinking cycle."""

    node_id: str
    state_snapshot: Any
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class WaveCycleRunner:
    """Minimal runner: records nodes for downstream TS-OS integration."""

    nodes: list[WaveCycleNode] = field(default_factory=list)

    def emit(self, node_id: str, state_snapshot: Any, **meta: Any) -> None:
        self.nodes.append(WaveCycleNode(node_id=node_id, state_snapshot=state_snapshot, meta=dict(meta)))

    def last(self) -> WaveCycleNode | None:
        return self.nodes[-1] if self.nodes else None


def attach_attractor_state(runner: WaveCycleRunner, label: str, tensor: Any) -> None:
    """Register an attractor hidden state under ``label``."""
    runner.emit(f"attractor::{label}", tensor)
