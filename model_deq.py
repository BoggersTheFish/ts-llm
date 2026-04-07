"""
DEQ (Deep Equilibrium) components for AttractorLM.

Theory
------
The fast relax step g(h) = h + α(tanh(W_ff h + b + W_fs h_slow + W_x x) - h)
is a contraction with unique fixed point h* satisfying:
    h* = tanh(W_ff h* + b + W_fs h_slow + W_x x)

Jacobian of g at h*:
    J = (1 - α)I + α · diag(1 - (h*)²) · W_ff
    (uses the fact that tanh(W_ff h* + ...) = h* at the fixed point)

Gradient strategy (hybrid IFT + BPTT)
---------------------------------------
Two distinct gradient flows need to be handled:

  1. PARAMETER gradients (W_ff, W_fs, W_x, W_ff_b, h_slow, x):
     These use the Implicit Function Theorem — how does h* move when a
     parameter shifts?  dL/dθ = v · (dg/dθ|_{h*}) where v solves
     (I - J)^T v = dL/dh*.  Solved by Tikhonov-regularised exact solve
     (robust when ρ(J) ≥ 1; O(d³), fine for d ≤ 256 on CPU).

  2. CROSS-STEP gradient (h_init → h_fast from the previous token):
     For a perfect contraction, h* is independent of h_init (dh*/dh_init→0),
     so the IFT gives ≈0 here — which would cut cross-token learning.
     Instead we apply K Jacobian steps: grad_h_init = J^T^K @ dL/dh*
     This matches what BPTT would compute through K relax steps, evaluated
     at the converged state rather than the intermediate states.

Parameter gradient derivations
--------------------------------
    dL/dW_ff   = α · outer(D * v, h*)    [where D = 1 - (h*)²]
    dL/dW_ff_b = α · D * v
    dL/dW_fs   = α · outer(D * v, h_slow)
    dL/dW_x    = α · outer(D * v, x)
    dL/dh_slow = α · W_fs^T @ (D * v)
    dL/dx      = α · W_x^T @ (D * v)

Neumann series (reference / diagnostics)
-----------------------------------------
neumann_solve() approximates (I-J)^{-T} v = g via:
    v ≈ Σ_{k=0}^{K} (J^T)^k g   (O(K d²), converges when ρ(J) < 1)
Prefer the exact solve in DEQFastRelax; Neumann is exposed for diagnostics.
"""

from __future__ import annotations

import torch
import torch.autograd


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def neumann_solve(
    J: torch.Tensor,
    g: torch.Tensor,
    steps: int = 50,
) -> torch.Tensor:
    """
    Approximate (I - J)^T v = g via the Neumann series:
        v ≈ Σ_{k=0}^{steps} (J^T)^k g

    Converges when spectral_radius(J) < 1.  O(steps × d²).
    Exposed for diagnostics; DEQFastRelax uses the exact solve instead.

    Steps needed for rel-error < ε:  steps ≥ log(ε) / log(ρ)
        ρ = 0.90 → ~44 steps for ε = 1e-2
        ρ = 0.75 → ~16 steps for ε = 1e-2
    """
    running = g.clone()
    acc = g.clone()
    for _ in range(steps):
        running = J.T @ running
        acc = acc + running
    return acc


def spectral_radius(J: torch.Tensor) -> float:
    """Largest singular value of J (upper bound on spectral radius)."""
    return float(torch.linalg.norm(J, ord=2).item())


def _tikhonov_solve(
    J: torch.Tensor,
    g: torch.Tensor,
    target_rho: float = 0.98,
) -> torch.Tensor:
    """
    Solve (I - J)^T v = g with Tikhonov regularisation.

    When ρ(J) ≥ target_rho we add ε·I to (I - J) to make it invertible:
        v = ((1 + ε)I - J)^{-T} g    where ε = max(0, ρ - target_rho) + 1e-6

    For ρ < target_rho: near-exact IFT.
    For ρ = 1.03: ε ≈ 0.05, condition number bounded by ~20.
    """
    rho = spectral_radius(J)
    eps = max(0.0, rho - target_rho) + 1e-6
    d = J.shape[0]
    I = torch.eye(d, device=J.device, dtype=J.dtype)
    A = (1.0 + eps) * I - J      # shifted: eigenvalues bounded away from zero
    return torch.linalg.solve(A.T, g)


# ---------------------------------------------------------------------------
# Custom autograd Function
# ---------------------------------------------------------------------------

class DEQFastRelax(torch.autograd.Function):
    """
    Fixed-point solver for the fast relax step with hybrid IFT / BPTT gradient.

    Call signature (all positional):
        DEQFastRelax.apply(
            h_init, h_slow, x,
            W_ff_w, W_ff_b, W_fs_w, W_x_fast_w,
            alpha,          # float
            max_steps,      # int  — solver iterations
            tol,            # float — convergence threshold (0 = always run max_steps)
            bptt_steps,     # int  — Jacobian steps for grad_h_init
        )

    Forward:
        Runs g(h) = h + α(tanh(W_ff h + b + W_fs h_slow + W_x x) - h)
        for up to max_steps until ||h_new - h|| < tol.

    Backward (hybrid):
        Parameter/conditioning grads — IFT via Tikhonov-regularised exact solve.
        Cross-step grad (h_init)    — K Jacobian steps: J^T^K @ dL/dh*.
    """

    @staticmethod
    def forward(
        ctx,
        h_init: torch.Tensor,
        h_slow: torch.Tensor,
        x: torch.Tensor,
        W_ff_w: torch.Tensor,
        W_ff_b: torch.Tensor,
        W_fs_w: torch.Tensor,
        W_x_fast_w: torch.Tensor,
        alpha: float,
        max_steps: int,
        tol: float,
        bptt_steps: int,
    ) -> torch.Tensor:
        with torch.no_grad():
            h = h_init.clone()
            for _ in range(max_steps):
                pre = W_ff_w @ h + W_ff_b + W_fs_w @ h_slow + W_x_fast_w @ x
                h_new = h + alpha * (torch.tanh(pre) - h)
                delta = torch.linalg.norm(h_new - h)
                h = h_new
                if tol > 0.0 and delta < tol:
                    break

            # Jacobian at the current (possibly approximate) fixed point
            pre_final = W_ff_w @ h + W_ff_b + W_fs_w @ h_slow + W_x_fast_w @ x
            D = 1.0 - torch.tanh(pre_final) ** 2   # sech² at h
            d = h.shape[0]
            I = torch.eye(d, device=h.device, dtype=h.dtype)
            J = (1.0 - alpha) * I + alpha * D.unsqueeze(1) * W_ff_w

        ctx.save_for_backward(h, J, D, h_slow, x, W_ff_w, W_fs_w, W_x_fast_w)
        ctx.alpha = alpha
        ctx.bptt_steps = bptt_steps
        return h.clone()

    @staticmethod
    def backward(ctx, grad_h_star: torch.Tensor):
        h_star, J, D, h_slow, x, W_ff_w, W_fs_w, W_x_fast_w = ctx.saved_tensors
        alpha = ctx.alpha
        bptt_steps = ctx.bptt_steps

        # --- IFT solve for parameter / conditioning gradients ---
        # v solves (I - J)^T v = dL/dh*  (regularised for safety when ρ ≥ 1)
        v = _tikhonov_solve(J, grad_h_star)
        aDv = alpha * D * v   # shape (d,)

        # --- Cross-step gradient: K Jacobian steps (matches BPTT at h*) ---
        # For a true contraction, dh*/dh_init → 0.  Instead we propagate
        # dL/dh* back through K applications of J^T, which equals what BPTT
        # would compute through K relax substeps evaluated at the fixed point.
        grad_h_init = grad_h_star.clone()
        for _ in range(bptt_steps):
            grad_h_init = J.T @ grad_h_init

        return (
            grad_h_init,                       # dL/dh_init  (cross-step, BPTT-style)
            W_fs_w.T @ aDv,                    # dL/dh_slow  (IFT)
            W_x_fast_w.T @ aDv,               # dL/dx       (IFT)
            torch.outer(aDv, h_star),          # dL/dW_ff_w  (IFT)
            aDv,                               # dL/dW_ff_b  (IFT)
            torch.outer(aDv, h_slow),          # dL/dW_fs_w  (IFT)
            torch.outer(aDv, x),              # dL/dW_x_w   (IFT)
            None,  # alpha
            None,  # max_steps
            None,  # tol
            None,  # bptt_steps
        )
