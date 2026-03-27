#!/usr/bin/env python3
"""Write 3D PCA of a short limit-cycle-style orbit (NumPy dynamics only)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _orbit_states(seed: int, dim: int, steps: int) -> np.ndarray:
    from attractor_llm.core import make_diffusion_matrix, step_state, text_to_signal

    rng = np.random.default_rng(seed)
    diff = make_diffusion_matrix(dim, rng)
    sig = text_to_signal("orbit demo", dim)
    s = np.zeros(dim)
    rows = [s.copy()]
    for _ in range(steps - 1):
        s = step_state(s, diff, sig, 0.05, cubic_scale=0.05)
        rows.append(s.copy())
    return np.stack(rows, axis=0)


def _pca3(x: np.ndarray) -> np.ndarray:
    """x: (T, D) -> (T, 3)"""
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    comp = vt[:3].T
    return x @ comp


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--out-png", type=Path, default=REPO_ROOT / "plots" / "limit_cycle_orbit.png")
    p.add_argument("--out-html", type=Path, default=REPO_ROOT / "plots" / "limit_cycle_orbit.html")
    args = p.parse_args()

    states = _orbit_states(args.seed, args.dim, args.steps)
    proj = _pca3(states)

    args.out_png.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib required for PNG") from e

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(proj[:, 0], proj[:, 1], proj[:, 2], color="#2a6f97", linewidth=1.5)
    ax.scatter(proj[-1, 0], proj[-1, 1], proj[-1, 2], color="#d62828", s=40, label="end")
    ax.set_title("PCA 3D — sandbox orbit (NumPy dynamics)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=140)
    plt.close(fig)

    try:
        import plotly.graph_objects as go
    except ImportError:
        args.out_html.write_text(
            "<!-- plotly not installed; install plotly to regenerate interactive HTML -->\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.out_png}; HTML skipped (no plotly)")
        return

    trace = go.Scatter3d(
        x=proj[:, 0],
        y=proj[:, 1],
        z=proj[:, 2],
        mode="lines+markers",
        line=dict(width=4, color="#2a6f97"),
        marker=dict(size=3),
    )
    figp = go.Figure(data=[trace])
    figp.update_layout(title="PCA 3D — limit-cycle orbit", margin=dict(l=0, r=0, t=40, b=0))
    figp.write_html(str(args.out_html), include_plotlyjs="cdn")
    print(f"Wrote {args.out_png} and {args.out_html}")


if __name__ == "__main__":
    main()
