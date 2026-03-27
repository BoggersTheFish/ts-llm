"""Phase 3 configuration contracts with default-off behavior.

Note:
    These dataclasses define optional controls only. They do not activate
    runtime Phase 3 behavior unless explicitly wired by future integration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class ConstraintConfig:
    """Configuration for deterministic generation-time constraint shaping."""

    enabled: bool = False
    max_repeat: int = 3
    repeat_penalty: float = 0.35


@dataclass(slots=True)
class SelfImproveConfig:
    """Configuration for detached training-time advisory behavior."""

    enabled: bool = False
    warmup_batches: int = 64
    window: int = 32
    strength: float = 0.10


@dataclass(slots=True)
class Phase3Config:
    """Container for all experimental Phase 3 controls."""

    constraints: ConstraintConfig
    self_improve: SelfImproveConfig
    metrics: bool = False

    @classmethod
    def disabled(cls) -> "Phase3Config":
        """Create a fully disabled Phase 3 configuration.

        Returns:
            Disabled ``Phase3Config`` instance.
        """
        return cls(constraints=ConstraintConfig(), self_improve=SelfImproveConfig(), metrics=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Phase3Config":
        """Build config from a possibly partial dictionary payload.

        Args:
            data: Optional raw mapping.

        Returns:
            Normalized ``Phase3Config`` with defaults.
        """
        if not data:
            return cls.disabled()
        c = data.get("constraints") if isinstance(data, dict) else None
        s = data.get("self_improve") if isinstance(data, dict) else None
        constraints = ConstraintConfig(
            enabled=bool((c or {}).get("enabled", False)),
            max_repeat=int((c or {}).get("max_repeat", 3)),
            repeat_penalty=float((c or {}).get("repeat_penalty", 0.35)),
        )
        self_improve = SelfImproveConfig(
            enabled=bool((s or {}).get("enabled", False)),
            warmup_batches=int((s or {}).get("warmup_batches", 64)),
            window=int((s or {}).get("window", 32)),
            strength=float((s or {}).get("strength", 0.10)),
        )
        return cls(constraints=constraints, self_improve=self_improve, metrics=bool(data.get("metrics", False)))

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a plain dictionary.

        Returns:
            Dictionary representation of the config dataclass tree.
        """
        return asdict(self)
