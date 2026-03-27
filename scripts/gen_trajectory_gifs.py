#!/usr/bin/env python3
"""Generate small GIFs of 2D projections of state trajectories for ts_attractor/README."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from ts_attractor.numpy_demo import integrate, make_diffusion_matrix, text_to_signal


def main() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    out_dir = REPO_ROOT / "ts_attractor" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    dim = 32
    rng = np.random.default_rng(0)
    diff = make_diffusion_matrix(dim, rng)
    sig = text_to_signal("rhythm", dim)
    s = np.zeros(dim)
    traj = [s.copy()]
    for _ in range(48):
        from ts_attractor.numpy_demo import step_state, stabilize_norm

        s = step_state(s, diff, sig, 0.05)
        s = stabilize_norm(s)
        traj.append(s.copy())
    arr = np.stack(traj)
    xy = arr[:, :2]

    fig, ax = plt.subplots(figsize=(4, 4))
    (line,) = ax.plot([], [], "b-", lw=1.2)
    ax.set_xlim(float(xy[:, 0].min()) - 0.1, float(xy[:, 0].max()) + 0.1)
    ax.set_ylim(float(xy[:, 1].min()) - 0.1, float(xy[:, 1].max()) + 0.1)
    ax.set_title("State trajectory (first 2 dims)")

    def init() -> tuple:
        line.set_data([], [])
        return (line,)

    def update(i: int) -> tuple:
        line.set_data(xy[: i + 1, 0], xy[: i + 1, 1])
        return (line,)

    ani = FuncAnimation(fig, update, frames=len(xy), init_func=init, blit=True, interval=50)
    out = out_dir / "trajectory_orbit.gif"
    ani.save(out, writer="pillow", fps=12)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
