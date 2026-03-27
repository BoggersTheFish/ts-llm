#!/usr/bin/env python3
"""Train attractor from YAML config (wraps ``run_attractor_llm``).

Without ``--execute``, prints the equivalent command only (no heavy training).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("pip install pyyaml") from e
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--execute", action="store_true", help="Run training (heavy; GPU expected)")
    args = p.parse_args()
    cfg = load_yaml(args.config)
    ckpt_dir = REPO_ROOT / "checkpoints" / "tinystories" / str(cfg.get("name", "run"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    meta = {"config": cfg, "seed": args.seed}
    (ckpt_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "run_attractor_llm.py"),
        "--mode",
        "train",
        "--dataset",
        "tinystories",
        "--tinystories-max-files",
        str(cfg.get("tinystories_max_files", 2)),
        "--tinystories-max-tokens",
        str(cfg.get("tinystories_max_tokens", 50_000)),
        "--state-dim",
        str(cfg.get("state_dim", 128)),
        "--epochs",
        str(cfg.get("epochs", 1)),
        "--batch-size",
        str(cfg.get("batch_size", 1)),
        "--device",
        "cuda",
        "--seed",
        str(args.seed),
    ]
    if not args.execute:
        print("Dry-run (no training). Equivalent command:")
        print(" ".join(cmd))
        print("Meta:", ckpt_dir / "run_meta.json")
        return
    subprocess.check_call(cmd, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
