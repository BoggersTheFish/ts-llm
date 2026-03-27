"""Phase 3 configuration contracts (all features default-off)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class ConstraintConfig:
    enabled: bool = False
    max_repeat: int = 3
    repeat_penalty: float = 0.35


@dataclass(slots=True)
class SelfImproveConfig:
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
        return cls(constraints=ConstraintConfig(), self_improve=SelfImproveConfig(), metrics=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Phase3Config":
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
        return asdict(self)
