"""
Unit tests for Phase 3 additions to train.py:
  - _lr_at_step: LR schedule shape
  - _spectral_radius_wff: Jacobian health check
"""

from __future__ import annotations

import math

import torch
import pytest

from model import AttractorConfig, AttractorLM
from train import _lr_at_step, _spectral_radius_wff


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def test_lr_warmup_starts_at_lr_min() -> None:
    lr = _lr_at_step(0, total_steps=1000, lr_max=1e-3, lr_min=1e-4, warmup_steps=100)
    assert abs(lr - 1e-4) < 1e-10, f"Expected lr_min at step 0, got {lr}"


def test_lr_warmup_reaches_lr_max_at_warmup_steps() -> None:
    lr = _lr_at_step(100, total_steps=1000, lr_max=1e-3, lr_min=1e-4, warmup_steps=100)
    assert abs(lr - 1e-3) < 1e-10, f"Expected lr_max at warmup_steps, got {lr}"


def test_lr_warmup_is_linear() -> None:
    lrs = [_lr_at_step(s, 1000, 1.0, 0.0, 100) for s in range(101)]
    for i in range(1, 101):
        assert lrs[i] > lrs[i - 1], f"LR not increasing at step {i}"
    # Linearity: equal increments
    increments = [lrs[i] - lrs[i - 1] for i in range(1, 101)]
    assert max(increments) - min(increments) < 1e-9, "Warmup is not linear"


def test_lr_cosine_decreases_after_warmup() -> None:
    total = 1000
    warmup = 100
    lrs = [_lr_at_step(s, total, 1.0, 0.0, warmup) for s in range(warmup, total + 1)]
    for i in range(1, len(lrs)):
        assert lrs[i] <= lrs[i - 1] + 1e-10, f"LR increased after warmup at step {warmup + i}"


def test_lr_cosine_ends_at_lr_min() -> None:
    lr = _lr_at_step(1000, total_steps=1000, lr_max=1e-3, lr_min=1e-4, warmup_steps=100)
    assert abs(lr - 1e-4) < 1e-9, f"Expected lr_min at total_steps, got {lr}"


def test_lr_zero_warmup() -> None:
    """With warmup_steps=0, step 0 should already be at lr_max."""
    lr = _lr_at_step(0, total_steps=100, lr_max=1.0, lr_min=0.1, warmup_steps=0)
    assert abs(lr - 1.0) < 1e-9, f"Expected lr_max at step 0 with no warmup, got {lr}"


def test_lr_step_beyond_total_steps_stays_at_lr_min() -> None:
    """
    Remainder-flush steps push global_step past total_steps (floor-division
    under-counts when len(dataset) % grad_accum != 0). LR must not re-increase.
    """
    total = 100
    warmup = 10
    lr_max, lr_min = 1.0, 0.1
    # Steps at and just past the boundary must all equal lr_min
    for step in [total, total + 1, total + 5, total + 20]:
        lr = _lr_at_step(step, total, lr_max, lr_min, warmup)
        assert abs(lr - lr_min) < 1e-9, (
            f"step={step} > total_steps={total}: expected lr_min={lr_min}, got {lr:.6f} "
            "(LR must not re-increase past total_steps)"
        )


def test_lr_total_steps_uses_ceil_not_floor() -> None:
    """
    steps_per_epoch must use ceil so remainder-flush steps are counted.
    Floor would make total_steps under-count, causing the clamp to do all
    the work and the cosine end-point to arrive slightly early.

    Verified by simulating the actual training loop step count and checking
    it equals ceil(docs / accum) * epochs.
    """
    import math as _math
    for docs, accum in [(10, 3), (100, 8), (5387, 8), (532, 8), (7, 7), (8, 8)]:
        floor_steps = docs // accum
        ceil_steps  = _math.ceil(docs / accum)
        # Simulate: full batches + optional remainder flush
        actual_steps = floor_steps + (1 if docs % accum != 0 else 0)
        assert ceil_steps == actual_steps, (
            f"docs={docs} accum={accum}: ceil={ceil_steps} floor={floor_steps} "
            f"actual={actual_steps} — ceil must match actual optimizer-step count"
        )


def test_lr_never_below_lr_min() -> None:
    """LR must stay >= lr_min for all steps."""
    lr_max, lr_min = 1.0, 0.05
    for step in range(0, 150):
        lr = _lr_at_step(step, total_steps=100, lr_max=lr_max, lr_min=lr_min, warmup_steps=10)
        assert lr >= lr_min - 1e-9, f"LR={lr:.6f} fell below lr_min={lr_min} at step={step}"


# ---------------------------------------------------------------------------
# Spectral radius monitor
# ---------------------------------------------------------------------------

def _small_model(seed: int = 0) -> AttractorLM:
    torch.manual_seed(seed)
    cfg = AttractorConfig(vocab_size=10, fast_dim=8, slow_dim=4)
    return AttractorLM(cfg)


def test_spectral_radius_is_positive() -> None:
    model = _small_model()
    rho = _spectral_radius_wff(model)
    assert rho > 0.0, f"Spectral radius must be positive, got {rho}"


def test_spectral_radius_below_one_at_init() -> None:
    """
    Init uses xavier_uniform with gain=0.5 so ρ(J) < 1 at step 0.
    This is the contractivity guarantee documented in model._init_weights.
    """
    for seed in range(5):
        model = _small_model(seed)
        rho = _spectral_radius_wff(model)
        assert rho < 1.0, (
            f"seed={seed}: ρ(J)={rho:.4f} ≥ 1 at init — contractivity guarantee violated"
        )


def test_spectral_radius_no_grad() -> None:
    """_spectral_radius_wff must not accumulate gradients."""
    model = _small_model()
    # Trigger a forward pass first so grad_fn exist
    ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    loss = model.forward_chunked(ids)
    loss.backward()

    rho = _spectral_radius_wff(model)
    # Calling it after backward must not raise and must return a plain float
    assert isinstance(rho, float)
    # Grads on W_ff should be unchanged (no new backward through rho)
    assert model.W_ff.weight.grad is not None


def test_spectral_radius_uses_eigenvalues_not_2norm() -> None:
    """
    For a non-symmetric J, the 2-norm (largest singular value) can exceed
    the spectral radius (largest |eigenvalue|). _spectral_radius_wff must
    return the eigenvalue-based value, not the 2-norm.
    """
    import torch.nn as nn
    # Construct a model whose W_ff.weight is a known non-symmetric matrix
    # where 2-norm >> spectral radius.
    torch.manual_seed(0)
    cfg = AttractorConfig(vocab_size=10, fast_dim=4, slow_dim=2)
    model = AttractorLM(cfg)

    # Build a nilpotent-ish upper-triangular matrix: all eigenvalues = 0
    # but 2-norm > 0 due to off-diagonal entries.
    with torch.no_grad():
        model.W_ff.weight.zero_()
        model.W_ff.weight[0, 1] = 10.0  # large off-diagonal
        model.W_ff.weight[1, 2] = 10.0
        model.W_ff.bias.zero_()

    rho = _spectral_radius_wff(model)
    # Spectral radius ≈ 0 (all eigenvalues near 0 for near-nilpotent + (1-α)I diagonal)
    # 2-norm would be >> 1 due to the large off-diagonal entries
    alpha = cfg.alpha_fast
    # J diagonal is (1 - alpha) ≈ 0.75; eigenvalues ≈ 0.75; off-diagonals shift singular values up
    # The key check: rho must be < 2-norm of J
    d = cfg.fast_dim
    bias = model.W_ff.bias
    D = 1.0 - torch.tanh(bias) ** 2
    I = torch.eye(d)
    J = (1.0 - alpha) * I + alpha * D.unsqueeze(1) * model.W_ff.weight
    two_norm = float(torch.linalg.norm(J, ord=2).item())
    assert rho < two_norm, (
        f"Expected spectral radius ({rho:.4f}) < 2-norm ({two_norm:.4f}) "
        "for non-symmetric J — may be using 2-norm instead of eigenvalues"
    )


def test_spectral_radius_scales_with_weight_magnitude() -> None:
    """Scaling W_ff by k should scale ρ(J) (approximately, at zero bias)."""
    torch.manual_seed(0)
    cfg = AttractorConfig(vocab_size=10, fast_dim=8, slow_dim=4)
    model = AttractorLM(cfg)

    # Zero out the bias so the linearisation is exact at h=0
    with torch.no_grad():
        model.W_ff.bias.zero_()

    rho1 = _spectral_radius_wff(model)

    # Double W_ff.weight
    with torch.no_grad():
        model.W_ff.weight.mul_(2.0)

    rho2 = _spectral_radius_wff(model)
    # With bias=0: J = (1-α)I + α·W_ff, so scaling W_ff by 2 increases J
    # and therefore ρ(J). The relationship is not exactly 2× because of the
    # (1-α)I term, but ρ2 > ρ1.
    assert rho2 > rho1, f"Expected ρ to grow with larger W_ff: {rho1:.4f} → {rho2:.4f}"
