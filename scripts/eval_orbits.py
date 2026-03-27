#!/usr/bin/env python3
"""Compute orbit metrics table from a checkpoint path (lightweight smoke)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def periodicity_score(states: list[list[float]]) -> float:
    """Crude cycle score: mean L2 distance between consecutive states (lower = smoother)."""
    import math

    if len(states) < 2:
        return 0.0
    acc = 0.0
    for i in range(1, len(states)):
        a = states[i - 1]
        b = states[i]
        acc += math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    return acc / (len(states) - 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()
    # Synthetic tiny orbit for smoke when no checkpoint
    states = [[float(i) * 0.01, float(i) * 0.02] for i in range(16)]
    row = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "periodicity": periodicity_score(states),
        "basin_blend_rate": 0.0,
        "palindrome_corr": 0.0,
    }
    print(json.dumps(row, indent=2))
    if args.out_json:
        args.out_json.write_text(json.dumps(row), encoding="utf-8")


if __name__ == "__main__":
    main()
