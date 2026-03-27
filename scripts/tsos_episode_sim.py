#!/usr/bin/env python3
"""Simulate 10 thinking episodes with orbit logging (TS-OS bridge stub)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from ts_attractor.tsos_bridge import WaveCycleRunner, attach_attractor_state


def main() -> None:
    log: list[dict] = []
    for ep in range(10):
        r = WaveCycleRunner()
        s = torch.randn(32)
        attach_attractor_state(r, f"ep{ep}", s)
        log.append({"episode": ep, "nodes": [n.node_id for n in r.nodes]})
    Path("tsos_episodes.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log[:2], indent=2))


if __name__ == "__main__":
    main()
