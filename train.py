#!/usr/bin/env python3
"""Minimal training entrypoint — equivalent to ``python run_attractor_llm.py --mode train ...``."""

from __future__ import annotations

import sys


def main() -> None:
    argv = [sys.argv[0], "--mode", "train"] + sys.argv[1:]
    sys.argv = argv
    from run_attractor_llm import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
