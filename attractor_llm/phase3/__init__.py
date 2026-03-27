"""Phase 3 isolated interfaces (spec-first, runtime-inactive by default).

Note:
    Importing this package does not activate Phase 3 runtime behavior.
"""

from attractor_llm.phase3.adapter import Phase3Adapter, Phase3ApplyResult, Phase3RuntimeState
from attractor_llm.phase3.config import ConstraintConfig, Phase3Config, SelfImproveConfig
from attractor_llm.phase3.contracts import Phase3Decision, Phase3MetricsSnapshot
from attractor_llm.phase3.controller import Phase3Controller
from attractor_llm.phase3.simulation import OfflineStepResult, run_offline_simulation, snapshots_from_dicts

__all__ = [
    "ConstraintConfig",
    "SelfImproveConfig",
    "Phase3Config",
    "Phase3MetricsSnapshot",
    "Phase3Decision",
    "Phase3Controller",
    "Phase3Adapter",
    "Phase3ApplyResult",
    "Phase3RuntimeState",
    "OfflineStepResult",
    "run_offline_simulation",
    "snapshots_from_dicts",
]

