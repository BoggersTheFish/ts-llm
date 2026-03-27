#!/usr/bin/env python3
"""Qualitative coherence scoring (NumPy toy; deterministic heuristic scores).

Target in the master plan (≥4.5) applies to full LM runs; this script stays lightweight.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ts_attractor.numpy_demo import toy_cycle_story


def _score_stream(ids: list[int]) -> float:
    """Heuristic 1–5 from local repetition structure."""
    if len(ids) < 2:
        return 3.0
    repeats = sum(1 for i in range(1, len(ids)) if ids[i] == ids[i - 1])
    r = repeats / max(len(ids) - 1, 1)
    return float(min(5.0, max(1.0, 3.0 + 2.0 * (1.0 - r))))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()
    prompts = ["A", "B", "C", "D", "E"]
    scores = []
    for seed, pr in enumerate(prompts):
        stream = toy_cycle_story(seed=seed + 10, length=40)
        sc = _score_stream(stream)
        scores.append({"prompt_type": pr, "score": sc})
    mean = sum(s["score"] for s in scores) / len(scores)
    out = {"mean": mean, "runs": scores}
    print(json.dumps(out, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(out), encoding="utf-8")


if __name__ == "__main__":
    main()
