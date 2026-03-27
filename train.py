#!/usr/bin/env python3
"""Convenience training entrypoint forwarding to main CLI.

Note:
    This wrapper only injects ``--mode train`` and does not change model math.
"""

from __future__ import annotations

import sys


def main() -> None:
    """Invoke main CLI in train mode.

    Returns:
        None.
    """
    argv = [sys.argv[0], "--mode", "train"] + sys.argv[1:]
    sys.argv = argv
    from run_attractor_llm import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
