"""Regression tests for the frozen eval harness (same training recipe as production demo)."""

from __future__ import annotations

import pytest

from eval_harness import (
    EVAL_SEED,
    MinimalAttractorLM,
    generate_with_cursor,
    generation_metrics,
    load_training_data,
    longest_repeated_ngram_length,
    metrics_passes_gates,
    train_and_evaluate,
)
from utils import build_vocab


def test_longest_repeated_ngram_length() -> None:
    assert longest_repeated_ngram_length([], max_ngram=5) == 0
    assert longest_repeated_ngram_length(["a"], max_ngram=5) == 0
    assert longest_repeated_ngram_length(["a", "b", "a", "b"], max_ngram=5) == 2
    assert longest_repeated_ngram_length(
        ["x", "y", "z", "x", "y", "z"], max_ngram=5
    ) == 3


def test_generation_metrics_exact_and_repeat() -> None:
    corpus = {"a b c", "x y z"}
    seqs = [["a", "b", "c"], ["a", "b", "a", "b"]]
    rate, longest = generation_metrics(seqs, corpus, max_ngram=5)
    assert rate == 0.5
    assert longest == 2
    r2, l2 = generation_metrics([["x", "y", "z", "x", "y", "z"]], corpus)
    assert r2 == 0.0
    assert l2 == 3
    assert generation_metrics([], corpus) == (0.0, 0)


def test_generate_with_cursor_smoke() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    corpus = ["the cat", "dog runs"]
    stoi, itos = build_vocab(corpus)
    model = MinimalAttractorLM(len(stoi), state_dim=8, relax_steps=1)
    device = torch.device("cpu")
    res = generate_with_cursor(
        model,
        stoi,
        itos,
        "the",
        max_len=4,
        device=device,
        k_repeat=3,
        track_attractors=False,
        temperature=0.0,
    )
    assert res.reason in ("max_steps", "eos", "repeating_window", "attractor_return")
    assert len(res.token_ids) >= 1


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
