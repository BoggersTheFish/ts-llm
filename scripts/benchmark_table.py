#!/usr/bin/env python3
"""Emit a Markdown benchmark table (placeholder metrics)."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    md = """# Attractor benchmark (stub)

| Model | CE (val) | periodicity | basin-blend | palindrome |
|-------|----------|-------------|-------------|------------|
| attractor_10m | TBD | TBD | TBD | TBD |
| attractor_50m | TBD | TBD | TBD | TBD |
| attractor_200m | TBD | TBD | TBD | TBD |
| transformer baseline | TBD | — | — | — |
| Hopfield baseline | TBD | — | — | — |

_Fill after GPU training runs on a capable machine._
"""
    Path("benchmark_table.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
