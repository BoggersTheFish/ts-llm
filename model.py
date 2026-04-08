"""
AttractorLM: multi-timescale attractor language model.

Two coupled hidden states replace the single flat h of MinimalAttractorLM:

  h_fast  — relaxes per-token toward a basin conditioned on (x, h_slow).
             Full BPTT within each chunk: no detach inside the relax loop.

  h_slow  — updates every token with a small alpha and a learned gate.
             Acts as a low-pass filter of h_fast's trajectory; carries
             discourse/topic context across chunk boundaries.

Token injection is gated: each token perturbs h_fast by a learned fraction
that depends on both the current state and the token embedding, so the model
can resist washing out earlier context when it matters.

Training: forward_chunked() truncates BPTT at chunk boundaries by detaching
both states before each chunk, while still carrying them forward. h_slow
therefore accumulates across the full document even though gradients are cut.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Defaults (change only when intentionally shifting the Sprint 1 baseline)
# ---------------------------------------------------------------------------

ALPHA_FAST: float = 0.25   # damping coefficient for fast relaxation
ALPHA_SLOW: float = 0.02   # damping coefficient for slow state update (~12× slower)
FAST_RELAX_STEPS: int = 2  # inner iterations per token in fast relaxation
BPTT_WINDOW: int = 16      # tokens per BPTT chunk (gradient truncation boundary)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AttractorConfig:
    vocab_size: int
    fast_dim: int = 32
    slow_dim: int = 16
    alpha_fast: float = ALPHA_FAST
    alpha_slow: float = ALPHA_SLOW
    fast_relax_steps: int = FAST_RELAX_STEPS
    # DEQ mode (Sprint 2): replaces the fixed-step relax loop with implicit
    # differentiation through the fixed point.  Set use_deq=True to enable.
    use_deq: bool = False
    deq_max_steps: int = 50  # solver iterations (forward pass)
    deq_tol: float = 1e-4    # convergence threshold (set 0 for fixed steps)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AttractorLM(nn.Module):
    """
    Multi-timescale attractor language model.

    Per-token step:
      1. Gated injection    — x perturbs h_fast by a learned context-dependent fraction
      2. Fast relaxation    — h_fast settles toward a basin conditioned on (x, h_slow)
      3. Slow state update  — h_slow accumulates a gated, damped version of h_fast's signal
      4. Readout            — logits = out(cat([h_fast, h_slow]))
    """

    def __init__(self, cfg: AttractorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        fd, sd = cfg.fast_dim, cfg.slow_dim

        # Token embedding
        self.embed = nn.Embedding(cfg.vocab_size, fd)

        # --- Gated injection ---
        # gate = sigmoid(W_gate_h(h_fast) + W_gate_x(x))
        # h_fast += gate * W_x_proj(x)
        self.W_gate_h = nn.Linear(fd, fd, bias=True)
        self.W_gate_x = nn.Linear(fd, fd, bias=False)
        self.W_x_proj = nn.Linear(fd, fd, bias=False)

        # --- Fast relaxation ---
        # target = tanh(W_ff(h_fast) + W_fs(h_slow) + W_x_fast(x))
        # h_fast += alpha_fast * (target - h_fast)
        self.W_ff = nn.Linear(fd, fd, bias=True)
        self.W_fs = nn.Linear(sd, fd, bias=False)
        self.W_x_fast = nn.Linear(fd, fd, bias=False)

        # --- Slow state gate + dynamics ---
        # slow_gate = sigmoid(W_sg_f(h_fast) + W_sg_s(h_slow))
        # slow_target = tanh(W_ss(h_slow) + W_sf(h_fast))
        # h_slow += alpha_slow * slow_gate * (slow_target - h_slow)
        self.W_sg_f = nn.Linear(fd, sd, bias=True)
        self.W_sg_s = nn.Linear(sd, sd, bias=False)
        self.W_ss = nn.Linear(sd, sd, bias=True)
        self.W_sf = nn.Linear(fd, sd, bias=False)

        # --- Readout ---
        self.out = nn.Linear(fd + sd, cfg.vocab_size, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Small init to keep fast relaxation contractive at t=0.

        With gain=0.5, Xavier gives W_ff spectral norm ≈ 0.5*sqrt(2) ≈ 0.7.
        The Jacobian of the fast map, J = (1-α)I + α*diag(sech²)*W_ff, then
        has ||J|| ≲ 0.75 + 0.25*0.7 ≈ 0.93 < 1 — Neumann series converges
        from step 0, which matters when we switch to DEQ in Sprint 2.
        """
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight, gain=0.5)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
        nn.init.normal_(self.embed.weight, std=0.02)

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(
        self,
        h_fast: torch.Tensor,
        h_slow: torch.Tensor,
        token_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process one token. Returns (logits, h_fast_new, h_slow_new).

        token_id: scalar long tensor on the same device as h_fast.

        Gradients flow through all fast relax substeps — no detach here.
        Detaching happens at chunk boundaries in forward_chunked().
        """
        x = self.embed(token_id)  # (fast_dim,)

        # 1. Gated injection
        gate = torch.sigmoid(self.W_gate_h(h_fast) + self.W_gate_x(x))
        h_fast = h_fast + gate * self.W_x_proj(x)

        # 2. Fast relaxation (conditioned on h_slow and x)
        if self.cfg.use_deq:
            from model_deq import DEQFastRelax
            h_fast = DEQFastRelax.apply(
                h_fast,
                h_slow,
                x,
                self.W_ff.weight,
                self.W_ff.bias,
                self.W_fs.weight,
                self.W_x_fast.weight,
                self.cfg.alpha_fast,
                self.cfg.deq_max_steps,
                self.cfg.deq_tol,
                self.cfg.fast_relax_steps,  # bptt_steps: K Jacobian steps for grad_h_init
            )
        else:
            # BPTT: fixed steps, gradients flow through the unrolled loop
            for _ in range(self.cfg.fast_relax_steps):
                target = torch.tanh(
                    self.W_ff(h_fast) + self.W_fs(h_slow) + self.W_x_fast(x)
                )
                h_fast = h_fast + self.cfg.alpha_fast * (target - h_fast)

        # 3. Slow state update (small alpha, gated)
        slow_gate = torch.sigmoid(self.W_sg_f(h_fast) + self.W_sg_s(h_slow))
        slow_target = torch.tanh(self.W_ss(h_slow) + self.W_sf(h_fast))
        h_slow = h_slow + self.cfg.alpha_slow * slow_gate * (slow_target - h_slow)

        # 4. Readout
        logits = self.out(torch.cat([h_fast, h_slow]))
        return logits, h_fast, h_slow

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def zero_state(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (h_fast, h_slow) initialised to zero."""
        return (
            torch.zeros(self.cfg.fast_dim, device=device),
            torch.zeros(self.cfg.slow_dim, device=device),
        )

    @torch.no_grad()
    def get_states_for_prefix(
        self,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run token_ids through the model and return the final (h_fast, h_slow).
        No gradient. Use for attractor geometry analysis.
        """
        device = token_ids.device
        h_fast, h_slow = self.zero_state(device)
        for t in range(token_ids.size(0)):
            _, h_fast, h_slow = self.step(h_fast, h_slow, token_ids[t])
        return h_fast.detach(), h_slow.detach()

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward_chunked(
        self,
        token_ids: torch.Tensor,
        chunk_size: int = BPTT_WINDOW,
        stop_grad_slow: bool = False,
    ) -> torch.Tensor:
        """
        Truncated BPTT over a single document.

        h_fast is always detached at chunk boundaries to bound gradient depth.
        h_slow is detached only when stop_grad_slow=True; by default gradients
        flow through h_slow across chunk boundaries so that W_ss/W_sf/W_sg_*
        receive signal from the full document, matching the accumulation
        behaviour seen at inference.

        Returns mean CE loss over all (T-1) next-token predictions.
        """
        device = token_ids.device
        T = token_ids.size(0)
        if T < 2:
            return torch.tensor(0.0, device=device)

        h_fast, h_slow = self.zero_state(device)
        losses: list[torch.Tensor] = []

        for start in range(0, T - 1, chunk_size):
            # Always cut h_fast gradient at chunk boundary
            h_fast = h_fast.detach()
            # Only cut h_slow gradient when explicitly requested
            if stop_grad_slow:
                h_slow = h_slow.detach()

            end = min(start + chunk_size, T - 1)
            chunk_logits: list[torch.Tensor] = []

            for t in range(start, end):
                logits, h_fast, h_slow = self.step(h_fast, h_slow, token_ids[t])
                chunk_logits.append(logits)

            logits_t = torch.stack(chunk_logits)        # (chunk_len, vocab)
            targets = token_ids[start + 1 : end + 1]   # next-token targets
            losses.append(F.cross_entropy(logits_t, targets))

        return torch.stack(losses).mean()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        token_ids: list[int],
        max_new: int,
        device: torch.device,
        *,
        temperature: float = 0.0,
        top_k: int | None = None,
    ) -> list[int]:
        """
        Autoregressive generation. Initialises state from the full prefix
        (all prefix tokens), then samples max_new new tokens.
        Returns the full sequence (prefix + generated).
        """
        self.eval()
        h_fast, h_slow = self.zero_state(device)
        ids = list(token_ids)

        # Prime state from full prefix; keep logits from the last step
        logits = torch.zeros(self.cfg.vocab_size, device=device)
        for tid in ids:
            t = torch.tensor(tid, dtype=torch.long, device=device)
            logits, h_fast, h_slow = self.step(h_fast, h_slow, t)

        # Generate
        for _ in range(max_new):
            next_id = _sample(logits, temperature=temperature, top_k=top_k)
            ids.append(next_id)
            t = torch.tensor(next_id, dtype=torch.long, device=device)
            logits, h_fast, h_slow = self.step(h_fast, h_slow, t)

        return ids


# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------

def _sample(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
) -> int:
    if temperature <= 0.0:
        return int(logits.argmax().item())
    scaled = logits / temperature
    if top_k is not None and top_k > 0 and top_k < scaled.numel():
        vals, idx = torch.topk(scaled, top_k)
        mask = torch.full_like(scaled, float("-inf"))
        mask.scatter_(0, idx, vals)
        scaled = mask
    return int(torch.multinomial(torch.softmax(scaled, dim=-1), 1).item())


# ---------------------------------------------------------------------------
# Quick smoke-run: python model.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from utils import build_vocab, encode, tokenize

    CORPUS = [
        "the cat chases the dog",
        "the dog runs to the barn",
        "the bird flies in the sky",
        "the fish swims in the lake",
        "the mouse eats the cheese",
    ]
    OVERSAMPLE = 4
    LR = 1e-2
    EPOCHS = 2000
    LOG_EVERY = 200

    torch.manual_seed(0)
    stoi, itos = build_vocab(CORPUS)
    cfg = AttractorConfig(vocab_size=len(stoi), fast_dim=32, slow_dim=16)
    model = AttractorLM(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # Oversampled training rows (mirrors frozen harness oversample logic)
    training_ids = [
        torch.tensor(encode(tokenize(s), stoi), dtype=torch.long)
        for s in CORPUS * OVERSAMPLE
    ]

    print(f"AttractorLM  fast_dim={cfg.fast_dim}  slow_dim={cfg.slow_dim}  "
          f"vocab={cfg.vocab_size}  params={sum(p.numel() for p in model.parameters())}")
    print(f"Training {EPOCHS} epochs on {len(CORPUS)} sentences × {OVERSAMPLE} oversample …\n")

    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        losses = [model.forward_chunked(ids) for ids in training_ids]
        torch.stack(losses).mean().backward()
        opt.step()

        if ep % LOG_EVERY == 0 or ep == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                eval_losses = [model.forward_chunked(ids).item() for ids in training_ids]
            mean_ce = sum(eval_losses) / len(eval_losses)
            print(f"epoch {ep:4d}  mean_CE={mean_ce:.4f}")

    # --- Branch tests (mirrors eval_harness branch_tests) ---
    branch_tests = [
        ("the cat", "chases"),
        ("the dog", "runs"),
        ("the bird", "flies"),
        ("the fish", "swims"),
        ("the mouse", "eats"),
    ]
    print("\nNext-token tests:")
    ok = 0
    model.eval()
    for prefix, expected in branch_tests:
        ids_t = torch.tensor(encode(tokenize(prefix), stoi), dtype=torch.long)
        h_fast, h_slow = model.zero_state(torch.device("cpu"))
        with torch.no_grad():
            for tid in ids_t:
                logits, h_fast, h_slow = model.step(h_fast, h_slow, tid)
        pred = itos[int(logits.argmax().item())]
        status = "ok" if pred == expected else "FAIL"
        ok += pred == expected
        print(f"  [{status}] {prefix!r} → {pred!r}  (expected {expected!r})")
    print(f"\n{ok}/{len(branch_tests)} next-token tests passed")

    # --- Attractor geometry: fast vs slow cosine distances ---
    print("\nAttractor geometry (pairwise cosine distances, fast states):")
    noun_prefixes = ["the cat", "the dog", "the bird", "the fish", "the mouse"]
    fast_states: dict[str, torch.Tensor] = {}
    slow_states: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for p in noun_prefixes:
            ids_p = torch.tensor(encode(tokenize(p), stoi), dtype=torch.long)
            hf, hs = model.get_states_for_prefix(ids_p)
            fast_states[p] = hf
            slow_states[p] = hs

    # Print fast-state pairwise cosines
    labels = [p.split()[-1] for p in noun_prefixes]
    w = max(len(l) for l in labels)
    header = " " * (w + 2) + "  ".join(f"{l:>{w}}" for l in labels)
    print(header)
    for i, pi in enumerate(noun_prefixes):
        row = f"{labels[i]:>{w}}"
        for j, pj in enumerate(noun_prefixes):
            fi = fast_states[pi].float()
            fj = fast_states[pj].float()
            cos = float(F.cosine_similarity(fi.unsqueeze(0), fj.unsqueeze(0)))
            row += f"  {cos:>{w}.3f}"
        print(row)

    # Slow state norms
    print("\nSlow state norms (should be non-trivially non-zero after training):")
    for p in noun_prefixes:
        norm = slow_states[p].norm().item()
        print(f"  {p!r:20s}  ||h_slow||={norm:.4f}")

    # --- Sample generation ---
    print("\nGreedy generation:")
    for prefix in ["the cat", "the dog", "the bird"]:
        prefix_ids = encode(tokenize(prefix), stoi)
        out_ids = model.generate(prefix_ids, max_new=6, device=torch.device("cpu"))
        print(f"  {prefix!r:15s} → {' '.join(itos[i] for i in out_ids)!r}")

    sys.exit(0 if ok == len(branch_tests) else 1)
