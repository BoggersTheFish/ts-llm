"""Regression tests for the frozen eval harness (same training recipe as production demo)."""

from __future__ import annotations

import pytest

from eval_harness import (
    EVAL_SEED,
    load_training_data,
    metrics_passes_gates,
    train_and_evaluate,
)


@pytest.mark.slow
def test_frozen_train_and_eval_passes_gates() -> None:
    """Full training + metrics must satisfy baseline gates (seed, architecture, corpus)."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(EVAL_SEED)
    _, _, _, _, m = train_and_evaluate(quiet=True)
    ok, reasons = metrics_passes_gates(m)
    assert ok, "; ".join(reasons)
    assert m.branch_correct == m.branch_total
    assert m.mean_corpus_ce <= load_training_data().gates.mean_ce_max
