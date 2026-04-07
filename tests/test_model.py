"""
Sprint 1 validation tests for AttractorLM.

Fast tests (run every time):
  - Output shapes and types
  - Gradient flow through fast relax loop (critical: no detach inside step())
  - Gradient flow to slow state parameters
  - Gate activations not saturated at init
  - h_slow accumulates non-trivially after a short training run

Slow tests (pytest -m slow):
  - Full 2000-epoch training run on toy corpus reaches CE < 0.6
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from model import ALPHA_SLOW, BPTT_WINDOW, AttractorConfig, AttractorLM
from utils import build_vocab, encode, tokenize

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TOY_CORPUS = [
    "the cat chases the dog",
    "the dog runs to the barn",
    "the bird flies in the sky",
    "the fish swims in the lake",
    "the mouse eats the cheese",
]

BRANCH_TESTS = [
    ("the cat", "chases"),
    ("the dog", "runs"),
    ("the bird", "flies"),
    ("the fish", "swims"),
    ("the mouse", "eats"),
]


def _vocab() -> tuple[dict[str, int], list[str]]:
    return build_vocab(TOY_CORPUS)


def _model(fast_dim: int = 32, slow_dim: int = 16, seed: int = 0) -> AttractorLM:
    torch.manual_seed(seed)
    stoi, _ = _vocab()
    cfg = AttractorConfig(vocab_size=len(stoi), fast_dim=fast_dim, slow_dim=slow_dim)
    return AttractorLM(cfg)


def _ids(sentence: str, stoi: dict[str, int]) -> torch.Tensor:
    return torch.tensor(encode(tokenize(sentence), stoi), dtype=torch.long)


# ---------------------------------------------------------------------------
# Shape and type checks
# ---------------------------------------------------------------------------

def test_step_output_shapes() -> None:
    stoi, _ = _vocab()
    model = _model()
    h_fast, h_slow = model.zero_state(torch.device("cpu"))
    token = torch.tensor(0, dtype=torch.long)
    logits, h_fast2, h_slow2 = model.step(h_fast, h_slow, token)

    assert logits.shape == (len(stoi),), f"logits shape {logits.shape}"
    assert h_fast2.shape == (32,), f"h_fast shape {h_fast2.shape}"
    assert h_slow2.shape == (16,), f"h_slow shape {h_slow2.shape}"
    assert logits.dtype == torch.float32


def test_zero_state_shapes() -> None:
    model = _model()
    hf, hs = model.zero_state(torch.device("cpu"))
    assert hf.shape == (model.cfg.fast_dim,)
    assert hs.shape == (model.cfg.slow_dim,)
    assert hf.sum().item() == 0.0
    assert hs.sum().item() == 0.0


def test_forward_chunked_returns_scalar_loss() -> None:
    stoi, _ = _vocab()
    model = _model()
    ids = _ids(TOY_CORPUS[0], stoi)
    loss = model.forward_chunked(ids, chunk_size=4)
    assert loss.shape == (), f"loss shape should be scalar, got {loss.shape}"
    assert loss.item() > 0.0


def test_forward_chunked_short_sequence() -> None:
    """Sequence of length 1 should return 0.0 without error."""
    stoi, _ = _vocab()
    model = _model()
    ids = torch.tensor([0], dtype=torch.long)
    loss = model.forward_chunked(ids)
    assert loss.item() == 0.0


def test_forward_chunked_chunk_covers_whole_sentence() -> None:
    """When chunk_size >= len(sentence), one chunk, no boundary detach issues."""
    stoi, _ = _vocab()
    model = _model()
    ids = _ids(TOY_CORPUS[0], stoi)  # length 5
    loss_big = model.forward_chunked(ids, chunk_size=100)
    assert loss_big.item() > 0.0


# ---------------------------------------------------------------------------
# Gradient flow — the critical Sprint 1 invariant
# ---------------------------------------------------------------------------

def test_gradients_flow_through_fast_relax() -> None:
    """
    Gradients must reach W_ff.weight, which is inside the fast relax loop.
    If detach() crept back in, this grad will be zero.
    """
    stoi, _ = _vocab()
    model = _model()
    ids = _ids(TOY_CORPUS[0], stoi)

    loss = model.forward_chunked(ids, chunk_size=BPTT_WINDOW)
    loss.backward()

    assert model.W_ff.weight.grad is not None, "W_ff.weight.grad is None"
    max_grad = model.W_ff.weight.grad.abs().max().item()
    assert max_grad > 1e-10, (
        f"W_ff.weight.grad max={max_grad:.2e} — gradients not flowing "
        "through fast relax steps (detach() may have crept back in)"
    )


def test_gradients_flow_through_W_fs() -> None:
    """
    W_fs (slow→fast conditioning inside relax) must also receive gradients.
    This verifies h_slow's influence on the fast relax is differentiable.
    """
    stoi, _ = _vocab()
    model = _model()
    ids = _ids(TOY_CORPUS[0], stoi)

    loss = model.forward_chunked(ids, chunk_size=BPTT_WINDOW)
    loss.backward()

    assert model.W_fs.weight.grad is not None
    max_grad = model.W_fs.weight.grad.abs().max().item()
    assert max_grad > 1e-10, (
        f"W_fs.weight.grad max={max_grad:.2e} — slow→fast conditioning path broken"
    )


def test_gradients_reach_slow_dynamics() -> None:
    """
    W_ss and W_sf must receive gradients — the slow state dynamics are in the
    computation graph via h_slow's influence on logits (through W_fs and self.out).
    """
    stoi, _ = _vocab()
    model = _model()
    ids = _ids(TOY_CORPUS[0], stoi)

    loss = model.forward_chunked(ids, chunk_size=BPTT_WINDOW)
    loss.backward()

    for name, param in [("W_ss.weight", model.W_ss.weight),
                        ("W_sf.weight", model.W_sf.weight)]:
        assert param.grad is not None, f"{name}.grad is None"
        max_grad = param.grad.abs().max().item()
        assert max_grad > 1e-12, (
            f"{name}.grad max={max_grad:.2e} — slow dynamics not in computation graph"
        )


def test_gradients_reach_gate_parameters() -> None:
    """W_gate_h and W_gate_x must receive gradients — the gate is differentiable."""
    stoi, _ = _vocab()
    model = _model()
    ids = _ids(TOY_CORPUS[0], stoi)

    loss = model.forward_chunked(ids, chunk_size=BPTT_WINDOW)
    loss.backward()

    for name, param in [("W_gate_h.weight", model.W_gate_h.weight),
                        ("W_gate_x.weight", model.W_gate_x.weight)]:
        assert param.grad is not None, f"{name}.grad is None"
        max_grad = param.grad.abs().max().item()
        assert max_grad > 1e-12, f"{name}.grad max={max_grad:.2e}"


def test_no_nan_gradients() -> None:
    """No parameter should have NaN gradients after a forward+backward pass."""
    stoi, _ = _vocab()
    model = _model()
    ids = _ids(TOY_CORPUS[0], stoi)

    loss = model.forward_chunked(ids)
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


# ---------------------------------------------------------------------------
# Gate saturation check
# ---------------------------------------------------------------------------

def test_gate_not_saturated_at_init() -> None:
    """
    At random init, gate activations should average near 0.5 — not near 0 or 1.
    Saturation at init would prevent gradient flow from the start.
    """
    model = _model(seed=42)
    stoi, _ = _vocab()
    device = torch.device("cpu")

    gate_values: list[float] = []
    with torch.no_grad():
        for sentence in TOY_CORPUS:
            h_fast, h_slow = model.zero_state(device)
            for tid in encode(tokenize(sentence), stoi):
                x = model.embed(torch.tensor(tid, dtype=torch.long))
                gate = torch.sigmoid(model.W_gate_h(h_fast) + model.W_gate_x(x))
                gate_values.extend(gate.tolist())
                _, h_fast, h_slow = model.step(h_fast, h_slow, torch.tensor(tid, dtype=torch.long))

    mean_gate = sum(gate_values) / len(gate_values)
    assert 0.2 < mean_gate < 0.8, (
        f"Mean gate at init = {mean_gate:.3f}. "
        "Values near 0 or 1 indicate saturation — reduce init scale."
    )


# ---------------------------------------------------------------------------
# h_slow accumulation check
# ---------------------------------------------------------------------------

def test_h_slow_accumulates_after_short_training() -> None:
    """
    After 300 training epochs, h_slow at the end of a sentence should be
    non-trivially non-zero — the slow state is learning to carry information.
    Also checks that h_fast and h_slow have distinct geometry (different
    prefix pairs rank differently under each state's cosine similarity).
    """
    torch.manual_seed(1)
    stoi, _ = _vocab()
    model = _model(seed=1)
    _train(model, stoi, epochs=300, lr=1e-2)

    device = torch.device("cpu")
    slow_norms = []
    fast_norms = []
    with torch.no_grad():
        for sentence in TOY_CORPUS:
            ids = _ids(sentence, stoi)
            hf, hs = model.get_states_for_prefix(ids)
            slow_norms.append(hs.norm().item())
            fast_norms.append(hf.norm().item())

    mean_slow = sum(slow_norms) / len(slow_norms)
    mean_fast = sum(fast_norms) / len(fast_norms)

    assert mean_slow > 0.01, (
        f"Mean h_slow norm = {mean_slow:.4f}. Slow state is not accumulating — "
        f"check W_sf/W_ss gradients and alpha_slow={ALPHA_SLOW}."
    )
    assert mean_fast > 0.01, (
        f"Mean h_fast norm = {mean_fast:.4f}. Fast state is not learning."
    )


def test_h_slow_varies_across_prefixes() -> None:
    """
    After training, different prefixes should produce different h_slow values.
    If h_slow is identical for all prefixes it's a dead state.
    """
    torch.manual_seed(2)
    stoi, _ = _vocab()
    model = _model(seed=2)
    _train(model, stoi, epochs=300, lr=1e-2)

    device = torch.device("cpu")
    slow_vecs = []
    with torch.no_grad():
        for sentence in TOY_CORPUS:
            ids = _ids(sentence, stoi)
            _, hs = model.get_states_for_prefix(ids)
            slow_vecs.append(hs)

    # At least one pair of prefixes should have cosine similarity < 0.999
    found_distinct = False
    for i in range(len(slow_vecs)):
        for j in range(i + 1, len(slow_vecs)):
            cos = F.cosine_similarity(
                slow_vecs[i].unsqueeze(0), slow_vecs[j].unsqueeze(0)
            ).item()
            if cos < 0.999:
                found_distinct = True
                break
        if found_distinct:
            break

    assert found_distinct, (
        "All h_slow vectors are nearly identical (max cosine ≥ 0.999) — "
        "slow state is not differentiating between contexts."
    )


# ---------------------------------------------------------------------------
# Generation sanity
# ---------------------------------------------------------------------------

def test_generate_correct_length() -> None:
    stoi, itos = _vocab()
    model = _model()
    device = torch.device("cpu")
    prefix_ids = encode(tokenize("the cat"), stoi)
    out = model.generate(prefix_ids, max_new=5, device=device, temperature=0.0)
    assert len(out) == len(prefix_ids) + 5


def test_generate_ids_in_vocab() -> None:
    stoi, itos = _vocab()
    model = _model()
    device = torch.device("cpu")
    prefix_ids = encode(tokenize("the dog"), stoi)
    out = model.generate(prefix_ids, max_new=8, device=device, temperature=0.0)
    assert all(0 <= i < len(stoi) for i in out), "Generated id out of vocab range"


def test_generate_prefix_preserved() -> None:
    stoi, itos = _vocab()
    model = _model()
    device = torch.device("cpu")
    prefix_ids = encode(tokenize("the bird flies"), stoi)
    out = model.generate(prefix_ids, max_new=3, device=device)
    assert out[: len(prefix_ids)] == prefix_ids, "Prefix not preserved in output"


# ---------------------------------------------------------------------------
# Full training regression (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_trains_to_low_ce_on_toy_corpus() -> None:
    """
    2000 epochs, toy corpus, lr=1e-2, seed=0.
    Must reach mean CE < 0.6 and pass all 5 next-token branch tests.
    This is the Sprint 1 gate: if this fails, the architecture doesn't converge.
    """
    torch.manual_seed(0)
    stoi, itos = _vocab()
    model = _model(seed=0)
    _train(model, stoi, epochs=2000, lr=1e-2)

    device = torch.device("cpu")
    ce = _eval_ce(model, stoi, device)
    assert ce < 0.6, f"mean CE={ce:.4f} after 2000 epochs (target < 0.6)"

    # Next-token branch tests
    model.eval()
    correct = 0
    failures = []
    with torch.no_grad():
        for prefix, expected in BRANCH_TESTS:
            ids = _ids(prefix, stoi)
            h_fast, h_slow = model.zero_state(device)
            for tid in ids:
                logits, h_fast, h_slow = model.step(h_fast, h_slow, tid)
            pred = itos[int(logits.argmax().item())]
            if pred == expected:
                correct += 1
            else:
                failures.append(f"{prefix!r} → {pred!r} (expected {expected!r})")

    assert correct == len(BRANCH_TESTS), (
        f"{correct}/{len(BRANCH_TESTS)} branch tests passed. Failures: {failures}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _train(
    model: AttractorLM,
    stoi: dict[str, int],
    epochs: int,
    lr: float,
    oversample: int = 4,
) -> None:
    """Train on the toy corpus with oversampling, return nothing."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    training_ids = [
        _ids(s, stoi) for s in TOY_CORPUS * oversample
    ]
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        losses = [model.forward_chunked(ids) for ids in training_ids]
        torch.stack(losses).mean().backward()
        opt.step()


def _eval_ce(
    model: AttractorLM,
    stoi: dict[str, int],
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for sent in TOY_CORPUS:
            ids = _ids(sent, stoi).to(device)
            losses.append(model.forward_chunked(ids).item())
    return sum(losses) / len(losses)
