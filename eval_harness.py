"""
Frozen evaluation harness: fixed seed, corpus, hyperparameters, and metrics.

Design stance: language state is a dynamical system that relaxes toward attractor
basins for semantic continuations—not attention over the full token history.
Run after any change: `python eval_harness.py` or `pytest tests/test_eval_harness.py`.

`run_demo` also prints extended attractor diagnostics (relax-until-convergence over probe
prefixes, stability, interpolation, verb/object/global-basin cosine tables); see README and
`docs/verified_run.md`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (
    build_vocab,
    decode,
    encode,
    first_divergence_index,
    shared_prefix_until_divergence,
    tokenize,
)

from state_analysis import labels_sorted, pairwise_cosine

# --- Frozen defaults (change these only when intentionally moving the baseline) ---

EVAL_SEED = 0
STATE_DIM = 32
RELAX_STEPS = 2
# Damped residual relax: h <- h + alpha * (tanh(W h + W_x x) - h); x = current token embed.
RELAX_ALPHA = 0.25
TRAIN_EPOCHS = 1500
LEARNING_RATE = 1e-2
# Oversample branched lines vs baseline (same recipe as main demo)
BRANCH_OVERSAMPLE = 4

_DEFAULT_TRAINING_PATH = Path(__file__).resolve().parent / "data" / "training.json"
_DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parent / "data" / "prompts.json"


@dataclass(frozen=True)
class EvalGates:
    """Optional thresholds from training.json `gates`; tune per corpus."""

    mean_ce_max: float = 0.35
    ambiguous_entropy_min: float | None = 0.4  # None = do not check ambiguous entropy


@dataclass(frozen=True)
class TrainingData:
    """Loaded from data/training.json — edit that file to change text without code changes."""

    corpus: tuple[str, ...]
    branch_line_count: int
    branch_tests: tuple[tuple[str, str], ...]
    ambiguous_prefix: str
    gates: EvalGates


@dataclass(frozen=True)
class PromptsData:
    """Loaded from data/prompts.json — prefixes for greedy decode experiments."""

    greedy_prefixes: tuple[str, ...]
    greedy_max_len: int


def training_data_path() -> Path:
    """Override with env TINYLLM_TRAINING_DATA=/path/to/training.json if needed."""
    override = os.environ.get("TINYLLM_TRAINING_DATA")
    return Path(override) if override else _DEFAULT_TRAINING_PATH


def prompts_data_path() -> Path:
    override = os.environ.get("TINYLLM_PROMPTS_DATA")
    return Path(override) if override else _DEFAULT_PROMPTS_PATH


def load_training_data(path: Path | None = None) -> TrainingData:
    p = path or training_data_path()
    if not p.is_file():
        raise FileNotFoundError(
            f"Training data file not found: {p}\n"
            "Create it or set TINYLLM_TRAINING_DATA to a JSON file (see data/training.json)."
        )
    raw = json.loads(p.read_text(encoding="utf-8"))
    corpus = tuple(str(s) for s in raw["corpus"])
    n_branch = int(raw["branch_line_count"])
    if not 0 <= n_branch <= len(corpus):
        raise ValueError("branch_line_count must be between 0 and len(corpus)")
    tests = tuple((str(a), str(b)) for a, b in raw["branch_tests"])
    amb = str(raw.get("ambiguous_prefix", ""))
    gr = raw.get("gates") or {}
    amb_min = gr.get("ambiguous_entropy_min", 0.4)
    if amb_min is not None:
        amb_min = float(amb_min)
    gates = EvalGates(
        mean_ce_max=float(gr.get("mean_ce_max", 0.35)),
        ambiguous_entropy_min=amb_min,
    )
    return TrainingData(
        corpus=corpus,
        branch_line_count=n_branch,
        branch_tests=tests,
        ambiguous_prefix=amb,
        gates=gates,
    )


def load_prompts(path: Path | None = None) -> PromptsData:
    p = path or prompts_data_path()
    if not p.is_file():
        raise FileNotFoundError(
            f"Prompts file not found: {p}\n"
            "Create data/prompts.json or set TINYLLM_PROMPTS_DATA."
        )
    raw = json.loads(p.read_text(encoding="utf-8"))
    prefixes = tuple(str(s) for s in raw["greedy_prefixes"])
    max_len = int(raw.get("greedy_max_len", 20))
    return PromptsData(greedy_prefixes=prefixes, greedy_max_len=max_len)


_TRAINING = load_training_data()
_PROMPTS = load_prompts()
CORPUS: list[str] = list(_TRAINING.corpus)
BRANCH_LINE_COUNT: int = _TRAINING.branch_line_count
BRANCH_TESTS: list[tuple[str, str]] = list(_TRAINING.branch_tests)
AMBIGUOUS_PREFIX: str = _TRAINING.ambiguous_prefix
EVAL_GATES: EvalGates = _TRAINING.gates


@dataclass(frozen=True)
class RelaxConvergenceResult:
    final_state: torch.Tensor
    num_steps: int
    norm_deltas: list[float]
    possible_limit_cycle: bool


def _cosine_consecutive(prev: torch.Tensor, curr: torch.Tensor) -> float:
    pf = prev.float().flatten()
    cf = curr.float().flatten()
    pn = torch.linalg.norm(pf)
    cn = torch.linalg.norm(cf)
    if float(pn) < 1e-12 or float(cn) < 1e-12:
        return float("nan")
    return float((pf @ cf) / (pn * cn))


def _sign3(c: float) -> int:
    if c > 1e-8:
        return 1
    if c < -1e-8:
        return -1
    return 0


def _limit_cycle_from_cosine_history(h_hist: list[torch.Tensor]) -> bool:
    """True if cos(h_t, h_{t-1}) flips sign repeatedly (possible 2-cycle)."""
    if len(h_hist) < 5:
        return False
    cos_seq: list[float] = []
    for i in range(1, len(h_hist)):
        cos_seq.append(_cosine_consecutive(h_hist[i - 1], h_hist[i]))
    signs = [_sign3(c) for c in cos_seq]
    flips = 0
    for j in range(1, len(signs)):
        if signs[j] != 0 and signs[j - 1] != 0 and signs[j] != signs[j - 1]:
            flips += 1
    return flips >= 3


class MinimalAttractorLM(nn.Module):
    """Recurrent state + relaxation toward fixed-point-style basins; no attention over history."""

    def __init__(
        self,
        vocab_size: int,
        state_dim: int = 16,
        relax_steps: int = 2,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.relax_steps = relax_steps
        self.embed = nn.Embedding(vocab_size, state_dim)
        self.W = nn.Linear(state_dim, state_dim, bias=True)
        self.W_x = nn.Linear(state_dim, state_dim, bias=False)
        self.out = nn.Linear(state_dim, vocab_size)

    def _relax_one_substep(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        new_h = torch.tanh(self.W(h) + self.W_x(x))
        return h + RELAX_ALPHA * (new_h - h)

    def relax(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.relax_steps):
            h = self._relax_one_substep(h, x)
            h = h.detach()
        return h

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        T = token_ids.size(0)
        device = token_ids.device
        h = torch.zeros(self.state_dim, device=device)
        logits_list: list[torch.Tensor] = []
        for t in range(T):
            x = self.embed(token_ids[t])
            h = h + x
            h = self.relax(h, x)
            logits_list.append(self.out(h))
        return torch.stack(logits_list, dim=0)

    @torch.no_grad()
    def get_state_for_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Same recurrence as forward (zero init, add embed, relax each step) but return
        the final relaxed hidden state instead of logits.
        """
        device = token_ids.device
        if token_ids.numel() == 0:
            return torch.zeros(self.state_dim, device=device).detach()
        h = torch.zeros(self.state_dim, device=device)
        for t in range(token_ids.size(0)):
            x = self.embed(token_ids[t])
            h = h + x
            h = self.relax(h, x)
        return h.detach()

    @torch.no_grad()
    def relax_until_convergence(
        self,
        h: torch.Tensor,
        x_embed: torch.Tensor,
        max_steps: int = 50,
        tol: float = 1e-4,
    ) -> RelaxConvergenceResult:
        """
        Repeatedly apply one inner damped relax step (same as inside `relax()`), with
        fixed token conditioning x_embed, track ||h_new - h_old||, stop at tol or max_steps.
        """
        device = h.device
        dtype = h.dtype
        h = h.reshape(-1).to(device=device, dtype=dtype).clone()
        x = x_embed.reshape(-1).to(device=device, dtype=dtype)
        deltas: list[float] = []
        h_hist: list[torch.Tensor] = [h.clone()]

        for step in range(max_steps):
            h_old = h.clone()
            h = self._relax_one_substep(h, x)
            h = h.detach()
            delta = float(torch.linalg.norm(h - h_old))
            deltas.append(delta)
            h_hist.append(h.clone())
            if delta < tol:
                return RelaxConvergenceResult(
                    final_state=h,
                    num_steps=step + 1,
                    norm_deltas=deltas,
                    possible_limit_cycle=_limit_cycle_from_cosine_history(h_hist),
                )

        return RelaxConvergenceResult(
            final_state=h,
            num_steps=max_steps,
            norm_deltas=deltas,
            possible_limit_cycle=_limit_cycle_from_cosine_history(h_hist),
        )

    @torch.no_grad()
    def trace_states(
        self,
        token_ids: torch.Tensor,
        itos: list[str] | None = None,
    ) -> list[tuple[str, torch.Tensor]]:
        """
        Hidden-state trajectory: start, after each token injection, after each relax substep.
        States are detached CPU tensors (1D float), matching forward/relax numerics.
        """
        device = token_ids.device
        traj: list[tuple[str, torch.Tensor]] = []
        h = torch.zeros(self.state_dim, device=device)
        traj.append(("start", h.detach().cpu().clone()))

        for t_idx in range(token_ids.size(0)):
            tid = int(token_ids[t_idx].item())
            tok_lbl = itos[tid] if itos is not None and tid < len(itos) else str(tid)
            x = self.embed(token_ids[t_idx])
            h = h + x
            traj.append((f"token:{tok_lbl}", h.detach().cpu().clone()))

            for r in range(self.relax_steps):
                h = self._relax_one_substep(h, x)
                h = h.detach()
                traj.append((f"relax:{r}", h.detach().cpu().clone()))

        return traj


def trace_prefix(
    model: MinimalAttractorLM,
    text: str,
    stoi: dict[str, int],
    device: torch.device | None = None,
) -> list[tuple[str, torch.Tensor]]:
    """Tokenize, encode, run `trace_states` on model device; labels use vocabulary words."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    ids = encode(tokenize(text), stoi)
    itos_list: list[str] = [""] * len(stoi)
    for w, i in stoi.items():
        itos_list[i] = w
    t = torch.tensor(ids, dtype=torch.long, device=device)
    return model.trace_states(t, itos=itos_list)


def print_trajectory_norm_report(traj: list[tuple[str, torch.Tensor]]) -> None:
    """Print L2 norms per step and cosine similarity to the previous step."""
    print("\nTrajectory norms:")
    print("step_name : ||state||  (cos to previous)")
    prev: torch.Tensor | None = None
    for name, hcpu in traj:
        nrm = float(torch.linalg.norm(hcpu.float()))
        if prev is None:
            print(f"{name:<18} norm={nrm:.4f}")
        else:
            c = _cosine_consecutive(prev, hcpu)
            cos_s = f"cos(prev)={c:.4f}" if c == c else "cos(prev)=nan"
            print(f"{name:<18} norm={nrm:.4f}  {cos_s}")
        prev = hcpu


@torch.no_grad()
def print_attractor_stability_probe(
    model: MinimalAttractorLM,
    prefixes: list[str],
    converged_np: dict[str, np.ndarray],
    stoi: dict[str, int],
    device: torch.device,
    *,
    n_perturb: int = 10,
    epsilon_std_scale: float = 0.05,
    relax_max_steps: int = 50,
    relax_tol: float = 1e-4,
    seed: int = EVAL_SEED,
) -> None:
    """
    Perturb each converged attractor with Gaussian noise (std = epsilon_std_scale * mean(|s|)),
    re-run relax_until_convergence with the same last-token embed, report recovery cosines and steps.
    """
    model.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    w_dtype = next(model.parameters()).dtype

    print("\nAttractor stability test:")
    print("# cos ≥ 0.95 → strong attractor basin")
    print("# cos 0.8–0.95 → weak basin")
    print("# cos < 0.8 → unstable basin")
    print()

    p_w = max(len("prefix"), max(len(p) for p in prefixes))
    c_mean, c_std, s_mean, c_min = 18, 18, 12, 18
    hdr = (
        f"{'prefix':<{p_w}}  "
        f"{'mean_cos_recovery':>{c_mean}}  "
        f"{'std_cos_recovery':>{c_std}}  "
        f"{'mean_steps':>{s_mean}}  "
        f"{'min_cos_recovery':>{c_min}}"
    )
    print(hdr)
    print("-" * len(hdr))

    for prefix in prefixes:
        s_orig = torch.tensor(
            converged_np[prefix], device=device, dtype=w_dtype
        ).reshape(-1)
        ids = encode(tokenize(prefix), stoi)
        tid = torch.tensor(ids, dtype=torch.long, device=device)
        x_last = model.embed(tid[-1]).reshape(-1).to(dtype=w_dtype)

        mean_abs = float(torch.mean(torch.abs(s_orig.float())))
        sigma = epsilon_std_scale * mean_abs
        if sigma < 1e-12:
            sigma = 1e-12

        cos_list: list[float] = []
        steps_list: list[int] = []
        for _ in range(n_perturb):
            eps = torch.randn(
                s_orig.shape,
                device=device,
                dtype=w_dtype,
                generator=gen,
            )
            h_pert = s_orig + eps * sigma
            rc = model.relax_until_convergence(
                h_pert, x_embed=x_last, max_steps=relax_max_steps, tol=relax_tol
            )
            c = _cosine_consecutive(s_orig, rc.final_state)
            cos_list.append(c)
            steps_list.append(rc.num_steps)

        ca = np.array(cos_list, dtype=np.float64)
        sa = np.array(steps_list, dtype=np.float64)
        mean_c = float(np.nanmean(ca))
        std_c = float(np.nanstd(ca, ddof=1)) if n_perturb > 1 else 0.0
        mean_st = float(np.nanmean(sa))
        min_c = float(np.nanmin(ca))

        mcs = f"{mean_c:.4f}" if mean_c == mean_c else "nan"
        scs = f"{std_c:.4f}" if std_c == std_c else "nan"
        mss = f"{mean_st:.2f}" if mean_st == mean_st else "nan"
        mins = f"{min_c:.4f}" if min_c == min_c else "nan"
        print(
            f"{prefix:<{p_w}}  {mcs:>{c_mean}}  {scs:>{c_std}}  {mss:>{s_mean}}  {mins:>{c_min}}"
        )


def _interpolation_display_tail(prefix: str) -> str:
    """Strip leading 'the ' for table titles and column labels."""
    if prefix.startswith("the "):
        return prefix[4:]
    return prefix


@torch.no_grad()
def print_attractor_interpolation_test(
    model: MinimalAttractorLM,
    prefix_a: str,
    prefix_b: str,
    converged_np: dict[str, np.ndarray],
    stoi: dict[str, int],
    device: torch.device,
    *,
    relax_max_steps: int = 50,
    relax_tol: float = 1e-4,
    verb_geometry: bool = False,
) -> None:
    """
    Linearly interpolate converged attractors, blend last-token embeds with the same t,
    relax each point with relax_until_convergence, report cosines to both endpoint attractors.
    """
    model.eval()
    w_dtype = next(model.parameters()).dtype

    s_a = torch.tensor(
        converged_np[prefix_a], device=device, dtype=w_dtype
    ).reshape(-1)
    s_b = torch.tensor(
        converged_np[prefix_b], device=device, dtype=w_dtype
    ).reshape(-1)

    tid_a = torch.tensor(
        encode(tokenize(prefix_a), stoi), dtype=torch.long, device=device
    )
    tid_b = torch.tensor(
        encode(tokenize(prefix_b), stoi), dtype=torch.long, device=device
    )
    x_a = model.embed(tid_a[-1]).reshape(-1).to(dtype=w_dtype)
    x_b = model.embed(tid_b[-1]).reshape(-1).to(dtype=w_dtype)

    tail_a = _interpolation_display_tail(prefix_a)
    tail_b = _interpolation_display_tail(prefix_b)
    title = f"{tail_a} → {tail_b}"
    col_a = f"cos(final, {tail_a})"
    col_b = f"cos(final, {tail_b})"
    tw = 6
    cw = max(20, len(col_a) + 1, len(col_b) + 1)

    print(f"\nAttractor interpolation test ({title})")
    print()
    hdr = f"{'t':<{tw}}  {col_a:>{cw}}  {col_b:>{cw}}"
    print(hdr)
    print("-" * len(hdr))

    t_values = [i / 10.0 for i in range(11)]
    for t in t_values:
        s_t = (1.0 - t) * s_a + t * s_b
        x_t = (1.0 - t) * x_a + t * x_b
        rc = model.relax_until_convergence(
            s_t, x_embed=x_t, max_steps=relax_max_steps, tol=relax_tol
        )
        final = rc.final_state.reshape(-1)
        c_a = _cosine_consecutive(s_a, final)
        c_b = _cosine_consecutive(s_b, final)
        sa = f"{c_a:.4f}" if c_a == c_a else "nan"
        sb = f"{c_b:.4f}" if c_b == c_b else "nan"
        print(f"{t:<{tw}.1f}  {sa:>{cw}}  {sb:>{cw}}")

    print("\nInterpretation:")
    if verb_geometry:
        print(
            "  • clean convergence toward one verb-prefix attractor → verb basins well separated"
        )
        print(
            "  • boundary region in t → verb basin boundary (or mixed conditioning)"
        )
        print(
            "  • distinct intermediate attractor → emergent state under blended token embed"
        )
    else:
        print(
            "  • if the system converges cleanly to cat or dog → basins are well separated"
        )
        print("  • if a boundary region appears → basin boundary detected")
        print("  • if a new attractor appears → emergent semantic state")


def collect_prefix_states(
    model: MinimalAttractorLM,
    prefixes: list[str],
    stoi: dict[str, int],
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Map each prefix string to its final relaxed state as a CPU numpy vector."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for p in prefixes:
            ids = encode(tokenize(p), stoi)
            t = torch.tensor(ids, dtype=torch.long, device=device)
            h = model.get_state_for_tokens(t)
            out[p] = h.cpu().numpy()
    return out


def make_training_ids(stoi: dict[str, int]) -> list[list[int]]:
    branch_rows = [encode(tokenize(s), stoi) for s in CORPUS[:BRANCH_LINE_COUNT]]
    baseline_rows = [encode(tokenize(s), stoi) for s in CORPUS[BRANCH_LINE_COUNT:]]
    return branch_rows * BRANCH_OVERSAMPLE + baseline_rows


def train_loop(
    model: MinimalAttractorLM,
    data: list[list[int]],
    epochs: int,
    *,
    lr: float = LEARNING_RATE,
    log_every: int = 100,
    quiet: bool = False,
) -> None:
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        losses: list[torch.Tensor] = []
        for row in data:
            seq = torch.tensor(row, dtype=torch.long, device=device)
            logits = model(seq)
            losses.append(F.cross_entropy(logits[:-1], seq[1:]))
        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()
        if not quiet and (ep % log_every == 0 or ep == epochs - 1):
            print(f"epoch {ep:4d}  loss {loss.item():.4f}")


@torch.no_grad()
def next_token_logits_after_prefix(
    model: MinimalAttractorLM, prefix_ids: list[int], device: torch.device
) -> torch.Tensor:
    if not prefix_ids:
        raise ValueError("prefix_ids must be non-empty")
    t = torch.tensor(prefix_ids, dtype=torch.long, device=device)
    logits = model(t)
    return logits[-1]


@torch.no_grad()
def top_k_tokens(
    logits: torch.Tensor, itos: list[str], k: int = 5
) -> list[tuple[str, float]]:
    probs = torch.softmax(logits, dim=-1)
    vals, idx = torch.topk(probs, k=min(k, logits.size(-1)))
    return [(itos[int(i)], float(v)) for v, i in zip(vals.tolist(), idx.tolist())]


@torch.no_grad()
def mean_corpus_cross_entropy(
    model: MinimalAttractorLM, data: list[list[int]], device: torch.device
) -> float:
    """Mean next-token CE across training sentences (eval mode)."""
    model.eval()
    losses: list[torch.Tensor] = []
    for row in data:
        seq = torch.tensor(row, dtype=torch.long, device=device)
        logits = model(seq)
        losses.append(F.cross_entropy(logits[:-1], seq[1:]))
    return float(torch.stack(losses).mean().item())


@torch.no_grad()
def generate(
    model: MinimalAttractorLM,
    stoi: dict[str, int],
    itos: list[str],
    prefix: str,
    max_len: int = 20,
    device: torch.device | None = None,
) -> str:
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    words = tokenize(prefix)
    ids = encode(words, stoi)
    h = torch.zeros(model.state_dim, device=device)
    out_ids = list(ids)
    for _ in range(max_len):
        x = model.embed(torch.tensor(out_ids[-1], device=device))
        h = h + x
        h = model.relax(h, x)
        logits = model.out(h)
        next_id = int(logits.argmax().item())
        out_ids.append(next_id)
    return " ".join(decode(out_ids, itos))


def entropy_from_logits(logits: torch.Tensor) -> float:
    p = torch.softmax(logits, dim=-1)
    log_p = torch.log(p.clamp_min(1e-12))
    return float(-(p * log_p).sum().item())


@dataclass
class EvalMetrics:
    mean_corpus_ce: float
    branch_correct: int
    branch_total: int
    branch_cases: list[dict[str, str | bool]] = field(default_factory=list)
    ambiguous_prefix: str = ""
    ambiguous_entropy: float = 0.0
    ambiguous_top5: list[tuple[str, float]] = field(default_factory=list)
    ambiguous_skipped: bool = False


@torch.no_grad()
def run_branch_and_ambiguous_eval(
    model: MinimalAttractorLM,
    stoi: dict[str, int],
    itos: list[str],
    device: torch.device,
) -> EvalMetrics:
    model.eval()
    cases: list[dict[str, str | bool]] = []
    ok = 0
    for prefix, expected in BRANCH_TESTS:
        pids = encode(tokenize(prefix), stoi)
        logits = next_token_logits_after_prefix(model, pids, device)
        pred = itos[int(logits.argmax().item())]
        match = pred == expected
        ok += int(match)
        cases.append(
            {"prefix": prefix, "expected": expected, "predicted": pred, "ok": match}
        )
    amb_str = AMBIGUOUS_PREFIX.strip()
    if amb_str:
        amb_ids = encode(tokenize(amb_str), stoi)
        alog = next_token_logits_after_prefix(model, amb_ids, device)
        amb_ent = entropy_from_logits(alog)
        amb_top5 = top_k_tokens(alog, itos, k=5)
        skipped = False
    else:
        amb_ent = float("nan")
        amb_top5 = []
        skipped = True
    return EvalMetrics(
        mean_corpus_ce=0.0,
        branch_correct=ok,
        branch_total=len(BRANCH_TESTS),
        branch_cases=cases,
        ambiguous_prefix=AMBIGUOUS_PREFIX,
        ambiguous_entropy=amb_ent,
        ambiguous_top5=amb_top5,
        ambiguous_skipped=skipped,
    )


def train_and_evaluate(
    *,
    device: torch.device | None = None,
    epochs: int = TRAIN_EPOCHS,
    seed: int = EVAL_SEED,
    quiet: bool = False,
) -> tuple[MinimalAttractorLM, dict[str, int], list[str], list[list[int]], EvalMetrics]:
    """Train with frozen recipe, return model, vocab, training rows, and metrics."""
    if device is None:
        device = torch.device("cpu")
    torch.manual_seed(seed)
    stoi, itos = build_vocab(CORPUS)
    train_ids = make_training_ids(stoi)
    vocab_size = len(stoi)
    model = MinimalAttractorLM(
        vocab_size=vocab_size, state_dim=STATE_DIM, relax_steps=RELAX_STEPS
    ).to(device)
    train_loop(model, train_ids, epochs=epochs, quiet=quiet)
    ce = mean_corpus_cross_entropy(model, train_ids, device)
    m = run_branch_and_ambiguous_eval(model, stoi, itos, device)
    m.mean_corpus_ce = ce
    return model, stoi, itos, train_ids, m


def metrics_passes_gates(m: EvalMetrics) -> tuple[bool, list[str]]:
    """Return (ok, failure reasons). Thresholds come from data/training.json `gates`."""
    reasons: list[str] = []
    g = _TRAINING.gates
    if m.branch_correct != m.branch_total:
        reasons.append(
            f"next-token tests {m.branch_correct}/{m.branch_total} (expected all correct)"
        )
    if m.mean_corpus_ce > g.mean_ce_max:
        reasons.append(f"mean_corpus_ce {m.mean_corpus_ce:.4f} > {g.mean_ce_max}")
    if (
        g.ambiguous_entropy_min is not None
        and not m.ambiguous_skipped
        and m.ambiguous_entropy < g.ambiguous_entropy_min
    ):
        reasons.append(
            f"ambiguous_entropy {m.ambiguous_entropy:.4f} < {g.ambiguous_entropy_min}"
        )
    return (len(reasons) == 0, reasons)


def print_metrics(m: EvalMetrics) -> None:
    print(f"mean_corpus_ce: {m.mean_corpus_ce:.6f}")
    print(f"next-token tests: {m.branch_correct}/{m.branch_total}")
    for c in m.branch_cases:
        status = "ok" if c["ok"] else "FAIL"
        print(
            f"  [{status}] {c['prefix']!r} -> {c['predicted']!r} (expected {c['expected']!r})"
        )
    print(f"ambiguous_prefix: {m.ambiguous_prefix!r}")
    if m.ambiguous_skipped:
        print("ambiguous_entropy: (skipped — empty ambiguous_prefix in training.json)")
    else:
        print(f"ambiguous_entropy: {m.ambiguous_entropy:.6f}")
        print(f"ambiguous_top5: {m.ambiguous_top5}")


def run_demo(*, include_greedy: bool = True, include_diagnostics: bool = True) -> int:
    """Train + print report; exit code 1 if gates fail."""
    print(
        "Frozen eval harness (seed=%d, state_dim=%d, epochs=%d)"
        % (EVAL_SEED, STATE_DIM, TRAIN_EPOCHS)
    )
    device = torch.device("cpu")
    torch.manual_seed(EVAL_SEED)
    stoi, itos = build_vocab(CORPUS)
    train_ids = make_training_ids(stoi)

    if include_diagnostics:
        if len(CORPUS) >= 2:
            wa = tokenize(CORPUS[0])
            wb = tokenize(CORPUS[1])
            print("Corpus pair (first two lines):", CORPUS[0], "|", CORPUS[1])
            print(f"  first_divergence_index (full strings)={first_divergence_index(wa, wb)!r}")
            tail_a, tail_b = wa[1:], wb[1:]
            if tail_a and tail_b:
                div_tail = first_divergence_index(tail_a, tail_b)
                shared_tail = shared_prefix_until_divergence(tail_a, tail_b)
                print(
                    f"  first_divergence_index (after first token)={div_tail!r} "
                    f"shared_prefix_until_divergence={shared_tail!r}"
                )
        else:
            print("Corpus (single line):", CORPUS[0] if CORPUS else "(empty)")
        print("\nTraining on:", CORPUS)

    vocab_size = len(stoi)
    model = MinimalAttractorLM(
        vocab_size=vocab_size, state_dim=STATE_DIM, relax_steps=RELAX_STEPS
    ).to(device)
    train_loop(model, train_ids, epochs=TRAIN_EPOCHS)

    m = run_branch_and_ambiguous_eval(model, stoi, itos, device)
    m.mean_corpus_ce = mean_corpus_cross_entropy(model, train_ids, device)

    print()
    print_metrics(m)

    noun_prefixes = [
        "the cat",
        "the dog",
        "the bird",
        "the fish",
        "the mouse",
    ]
    verb_prefixes = [
        "the cat chases",
        "the dog runs",
        "the bird flies",
        "the fish swims",
        "the mouse eats",
    ]
    # Same verb, different subjects — shared-basin diagnostic (order = matrix rows/cols).
    verb_basin_comparison_prefixes = [
        "the cat chases",
        "the dog chases",
        "the mouse chases",
        "the dog runs",
        "the cat runs",
    ]
    # Object after verb phrase — object-basin diagnostic (order = matrix rows/cols).
    object_basin_comparison_prefixes = [
        "the cat chases the dog",
        "the cat chases the mouse",
        "the cat chases the bird",
        "the mouse eats the cheese",
        "the mouse eats the dog",
    ]
    # Same final object, different verb paths — cross-verb object basin diagnostic.
    global_object_basin_comparison_prefixes = [
        "the cat chases the dog",
        "the mouse eats the dog",
        "the bird flies to the dog",
        "the cat chases the mouse",
        "the bird flies to the mouse",
    ]
    _seen_relax: set[str] = set()
    relax_prefixes: list[str] = []
    for p in (
        noun_prefixes
        + verb_prefixes
        + verb_basin_comparison_prefixes
        + object_basin_comparison_prefixes
        + global_object_basin_comparison_prefixes
    ):
        if p not in _seen_relax:
            _seen_relax.add(p)
            relax_prefixes.append(p)
    state_dict = collect_prefix_states(model, noun_prefixes, stoi, device)
    cos_dist = pairwise_cosine(state_dict)
    labels = labels_sorted(state_dict)
    cos_sim = 1.0 - cos_dist
    print("\nPairwise cosine similarity between animal states:")
    col_w = max(10, max(len(s) for s in labels) + 1)
    header = " " * col_w + "".join(f"{lab:>{col_w}}" for lab in labels)
    print(header)
    for i, li in enumerate(labels):
        row = "".join(f"{cos_sim[i, j]:>{col_w}.4f}" for j in range(len(labels)))
        print(f"{li:<{col_w}}{row}")

    traj_demo = trace_prefix(model, "the cat chases the", stoi, device)
    print_trajectory_norm_report(traj_demo)

    print("\nRelax-until-convergence (extra dynamics on prefix final states):")
    converged_np: dict[str, np.ndarray] = {}
    forward_np: dict[str, np.ndarray] = {}
    for prefix in relax_prefixes:
        ids = encode(tokenize(prefix), stoi)
        tid = torch.tensor(ids, dtype=torch.long, device=device)
        h_forward = model.get_state_for_tokens(tid)
        x_last = model.embed(tid[-1])
        rc = model.relax_until_convergence(
            h_forward, x_embed=x_last, max_steps=50, tol=1e-4
        )
        fnorm = float(torch.linalg.norm(rc.final_state.float()))
        converged_np[prefix] = rc.final_state.cpu().numpy()
        if prefix in noun_prefixes:
            forward_np[prefix] = h_forward.cpu().numpy()
        print(prefix)
        print(f"  steps_to_converge={rc.num_steps}")
        print(f"  final_norm={fnorm:.4f}")
        if rc.possible_limit_cycle:
            print("  possible limit cycle")

    print_attractor_stability_probe(
        model, noun_prefixes, converged_np, stoi, device
    )

    print_attractor_interpolation_test(
        model, "the cat", "the dog", converged_np, stoi, device
    )
    print_attractor_interpolation_test(
        model,
        "the cat chases",
        "the dog runs",
        converged_np,
        stoi,
        device,
        verb_geometry=True,
    )

    converged_noun = {k: converged_np[k] for k in noun_prefixes}
    conv_labels = labels_sorted(converged_noun)
    cd_conv = pairwise_cosine(converged_noun)
    sim_conv = 1.0 - cd_conv
    cw = max(12, max(len(s) for s in conv_labels) + 2)
    print("\nPairwise cosine similarity between converged attractor states:")
    hdr = " " * cw + "".join(f"{lab:>{cw}}" for lab in conv_labels)
    print(hdr)
    for i, li in enumerate(conv_labels):
        row = "".join(f"{sim_conv[i, j]:>{cw}.4f}" for j in range(len(conv_labels)))
        print(f"{li:<{cw}}{row}")

    converged_verb = {k: converged_np[k] for k in verb_prefixes}
    v_labels = labels_sorted(converged_verb)
    cd_v = pairwise_cosine(converged_verb)
    sim_v = 1.0 - cd_v
    vw = max(12, max(len(s) for s in v_labels) + 2)
    print(
        "\nPairwise cosine similarity between verb-conditioned attractor states:"
    )
    v_hdr = " " * vw + "".join(f"{lab:>{vw}}" for lab in v_labels)
    print(v_hdr)
    for i, li in enumerate(v_labels):
        row = "".join(f"{sim_v[i, j]:>{vw}.4f}" for j in range(len(v_labels)))
        print(f"{li:<{vw}}{row}")

    basin_states = {k: converged_np[k] for k in verb_basin_comparison_prefixes}
    b_labels = list(verb_basin_comparison_prefixes)
    Xb = np.stack([basin_states[k] for k in b_labels], axis=0).astype(
        np.float64, copy=False
    )
    bn = np.maximum(np.linalg.norm(Xb, axis=1, keepdims=True), 1e-12)
    Xbn = Xb / bn
    sim_b = Xbn @ Xbn.T
    bw = max(12, max(len(s) for s in b_labels) + 2)
    print("\nVerb basin comparison:")
    b_hdr = " " * bw + "".join(f"{lab:>{bw}}" for lab in b_labels)
    print(b_hdr)
    for i, li in enumerate(b_labels):
        row = "".join(f"{sim_b[i, j]:>{bw}.4f}" for j in range(len(b_labels)))
        print(f"{li:<{bw}}{row}")

    def _pair_cos(p1: str, p2: str) -> str:
        t1 = torch.from_numpy(converged_np[p1]).float()
        t2 = torch.from_numpy(converged_np[p2]).float()
        cc = _cosine_consecutive(t1, t2)
        return f"{cc:.4f}" if cc == cc else "nan"

    print()
    print(f'  cos("the cat chases", "the dog chases")={_pair_cos("the cat chases", "the dog chases")}')
    print(
        f'  cos("the cat chases", "the mouse chases")={_pair_cos("the cat chases", "the mouse chases")}'
    )
    print(f'  cos("the dog runs", "the cat runs")={_pair_cos("the dog runs", "the cat runs")}')

    object_states = {k: converged_np[k] for k in object_basin_comparison_prefixes}
    o_labels = list(object_basin_comparison_prefixes)
    Xo = np.stack([object_states[k] for k in o_labels], axis=0).astype(
        np.float64, copy=False
    )
    on = np.maximum(np.linalg.norm(Xo, axis=1, keepdims=True), 1e-12)
    Xon = Xo / on
    sim_o = Xon @ Xon.T
    ow = max(12, max(len(s) for s in o_labels) + 2)
    print("\nObject basin comparison:")
    print("cosine similarity matrix between these converged states.")
    o_hdr = " " * ow + "".join(f"{lab:>{ow}}" for lab in o_labels)
    print(o_hdr)
    for i, li in enumerate(o_labels):
        row = "".join(f"{sim_o[i, j]:>{ow}.4f}" for j in range(len(o_labels)))
        print(f"{li:<{ow}}{row}")

    print()
    print(
        f'  cos("the cat chases the dog", "the cat chases the mouse")={_pair_cos("the cat chases the dog", "the cat chases the mouse")}'
    )
    print(
        f'  cos("the cat chases the dog", "the cat chases the bird")={_pair_cos("the cat chases the dog", "the cat chases the bird")}'
    )
    print(
        f'  cos("the mouse eats the cheese", "the mouse eats the dog")={_pair_cos("the mouse eats the cheese", "the mouse eats the dog")}'
    )

    g_states = {k: converged_np[k] for k in global_object_basin_comparison_prefixes}
    g_labels = list(global_object_basin_comparison_prefixes)
    Xg = np.stack([g_states[k] for k in g_labels], axis=0).astype(
        np.float64, copy=False
    )
    gn = np.maximum(np.linalg.norm(Xg, axis=1, keepdims=True), 1e-12)
    Xgn = Xg / gn
    sim_g = Xgn @ Xgn.T
    gw = max(12, max(len(s) for s in g_labels) + 2)
    print("\nGlobal object basin comparison:")
    print("cosine similarity matrix between these states.")
    g_hdr = " " * gw + "".join(f"{lab:>{gw}}" for lab in g_labels)
    print(g_hdr)
    for i, li in enumerate(g_labels):
        row = "".join(f"{sim_g[i, j]:>{gw}.4f}" for j in range(len(g_labels)))
        print(f"{li:<{gw}}{row}")

    print()
    print(
        f'  cos("the cat chases the dog", "the mouse eats the dog")={_pair_cos("the cat chases the dog", "the mouse eats the dog")}'
    )
    print(
        f'  cos("the cat chases the dog", "the bird flies to the dog")={_pair_cos("the cat chases the dog", "the bird flies to the dog")}'
    )
    print(
        f'  cos("the cat chases the mouse", "the bird flies to the mouse")={_pair_cos("the cat chases the mouse", "the bird flies to the mouse")}'
    )

    print(
        "\nNoun vs verb converged attractor states (same subject; verb extends prefix):"
    )
    c_cat = _cosine_consecutive(
        torch.from_numpy(converged_np["the cat"]).float(),
        torch.from_numpy(converged_np["the cat chases"]).float(),
    )
    c_dog = _cosine_consecutive(
        torch.from_numpy(converged_np["the dog"]).float(),
        torch.from_numpy(converged_np["the dog runs"]).float(),
    )
    cat_cos_s = f"{c_cat:.4f}" if c_cat == c_cat else "nan"
    dog_s = f"{c_dog:.4f}" if c_dog == c_dog else "nan"
    print(f'  cos("the cat", "the cat chases")={cat_cos_s}')
    print(f'  cos("the dog", "the dog runs")={dog_s}')
    print(
        f'\ncos(state("the cat"), state("the cat chases"))={cat_cos_s} '
        "(converged attractor states)"
    )

    print(
        "\nForward pass hidden state (after "
        f"{RELAX_STEPS} relax substeps per token) vs converged attractor:"
    )
    for prefix in noun_prefixes:
        hf = torch.from_numpy(forward_np[prefix]).float()
        hc = torch.from_numpy(converged_np[prefix]).float()
        c = _cosine_consecutive(hf, hc)
        cos_s = f"{c:.4f}" if c == c else "nan"
        print(prefix)
        print(f"  cos(prefix_state , attractor_state)={cos_s}")

    if include_greedy:
        print("\nGreedy generations (prefixes from data/prompts.json):")
        for p in _PROMPTS.greedy_prefixes:
            if not p or not str(p).strip():
                continue
            text = generate(
                model, stoi, itos, p, max_len=_PROMPTS.greedy_max_len, device=device
            )
            print(f"  prefix={p!r} -> {text}")

    ok, reasons = metrics_passes_gates(m)
    if not ok:
        print("\nGATE FAILURES:")
        for r in reasons:
            print(" ", r)
        return 1
    print("\nAll gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_demo())
