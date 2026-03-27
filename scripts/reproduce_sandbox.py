#!/usr/bin/env python3
"""Reproducibility check for NumPy sandbox dynamics (Phase 0).

Default: compares a fresh trajectory to a committed golden JSON (fast math after import).

Use ``--full-dim 512`` to match the large golden fixture. Omit for quick (dim 64) check.

Heavy full-LM training reproduction is intentionally not run here; see ``docs/CHANGELOG.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _trajectory(seed: int, dim: int, steps: int) -> tuple[list[float], float]:
    from attractor_llm.core import make_diffusion_matrix, step_state, text_to_signal

    rng = np.random.default_rng(seed)
    diff = make_diffusion_matrix(dim, rng)
    sig = text_to_signal("sandbox", dim)
    s = np.zeros(dim)
    curve: list[float] = []
    for _ in range(steps):
        s = step_state(s, diff, sig, 0.05, cubic_scale=0.05)
        curve.append(float(np.linalg.norm(s)))
    return curve, curve[-1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dim", type=int, default=64, help="State dimension (64=quick, 512=full golden)")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument(
        "--golden",
        type=Path,
        default=None,
        help="Path to golden JSON (defaults by --dim)",
    )
    p.add_argument("--rtol", type=float, default=1e-9)
    p.add_argument("--atol", type=float, default=1e-6)
    args = p.parse_args()

    golden_path = args.golden
    if golden_path is None:
        name = "reproduce_sandbox_golden.json" if args.dim >= 512 else "reproduce_sandbox_golden_quick.json"
        golden_path = REPO_ROOT / "tests" / "fixtures" / name

    if not golden_path.is_file():
        print(f"Missing golden file: {golden_path}", file=sys.stderr)
        return 2

    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    curve, final = _trajectory(args.seed, args.dim, args.steps)

    exp_curve = golden["norm_curve"]
    if len(exp_curve) != len(curve):
        print(f"Step count mismatch: golden {len(exp_curve)} vs now {len(curve)}", file=sys.stderr)
        return 1

    ok = np.allclose(np.array(curve), np.array(exp_curve), rtol=args.rtol, atol=args.atol)
    if not ok:
        max_err = float(np.max(np.abs(np.array(curve) - np.array(exp_curve))))
        print(f"Mismatch vs {golden_path}; max abs err={max_err}", file=sys.stderr)
        return 1

    print(f"OK: trajectory matches {golden_path.name} (final_norm={final:.12f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
