"""
Minimal attractor LM — demo entrypoint.

State evolves by relaxation dynamics toward continuation basins (no transformer-style
attention over prior tokens). Frozen metrics and training recipe: `eval_harness.py`.
"""

from eval_harness import run_demo

if __name__ == "__main__":
    raise SystemExit(run_demo())
