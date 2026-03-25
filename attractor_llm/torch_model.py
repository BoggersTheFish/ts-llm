"""PyTorch trainable attractor language model — CE loss from negative distance to learned attractors."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from attractor_llm.embeddings import LearnableProtoEmbedder
from attractor_llm.model import DEFAULT_VOCAB
from attractor_llm.torch_core import AttractorDynamics

if TYPE_CHECKING:
    from attractor_llm.tokenizer import AttractorTokenizer


class TorchAttractorLanguageModel(nn.Module):
    r"""
    Trainable attractor LM: internal state follows :class:`AttractorDynamics`; next-token
    logits are

    .. math::

        \ell_v(s) = -\frac{\|s - a_v\|_2}{\tau},

    where :math:`a_v` is the learned attractor for token :math:`v` (fixed-step flow from
    zero under signal :math:`u_v`), and :math:`\tau` is a learnable temperature.
    """

    def __init__(
        self,
        state_dim: int = 128,
        vocab: list[str] | None = None,
        *,
        tokenizer: AttractorTokenizer | None = None,
        cubic_scale: float = 0.05,
        dt: float = 0.05,
        num_attractor_steps: int = 16,
        num_converge_steps: int = 12,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.num_attractor_steps = num_attractor_steps
        self.num_converge_steps = num_converge_steps
        self.tokenizer: AttractorTokenizer | None = tokenizer

        if tokenizer is not None:
            vs = tokenizer.get_vocab_size()
            self._vocab_list: list[str] = list(vocab) if vocab is not None else []
            self.embedder = LearnableProtoEmbedder(dim=state_dim, vocab=self._vocab_list, vocab_size=vs)
        else:
            self._vocab_list = list(vocab) if vocab is not None else list(DEFAULT_VOCAB)
            self.embedder = LearnableProtoEmbedder(dim=state_dim, vocab=self._vocab_list)

        self.dynamics = AttractorDynamics(dim=state_dim, cubic_scale=cubic_scale, dt=dt)
        self.temperature = nn.Parameter(torch.tensor(0.1))

        self.register_buffer(
            "_attractors_cache",
            torch.zeros(self.embedder.vocab_size, state_dim),
        )

    @property
    def vocab(self) -> list[str]:
        """Word labels when using toy vocab; may be empty when using subword-only mode."""
        return self._vocab_list

    def _tau(self) -> torch.Tensor:
        return self.temperature.clamp(min=1e-6)

    def logits_from_state(
        self,
        state: torch.Tensor,
        attractors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        state :
            ``(D,)`` or ``(batch, D)`` converged state.
        attractors :
            ``(V, D)`` precomputed proto-attractors.

        Returns
        -------
        torch.Tensor
            Logits ``(batch, V)`` with :math:`\\ell = -\\|s - a_v\\| / \\tau`.
        """
        if state.ndim == 1:
            state_b = state.unsqueeze(0)
        else:
            state_b = state
        dist = torch.cdist(state_b, attractors)
        return -dist / self._tau()

    def training_step(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        num_converge_steps: int | None = None,
        num_attractor_steps: int | None = None,
    ) -> torch.Tensor:
        """
        Teacher-forced sequence loss: for each position :math:`t`, inject
        ``input_ids[t]``, integrate, then predict ``target_ids[t]`` with cross-entropy.

        Parameters
        ----------
        input_ids, target_ids :
            1D tensors of length ``L`` on the same device as parameters.
        """
        if input_ids.ndim != 1 or target_ids.ndim != 1:
            raise ValueError("training_step expects 1D input_ids and target_ids")
        if input_ids.shape[0] != target_ids.shape[0]:
            raise ValueError("input_ids and target_ids must have the same length")

        device = input_ids.device
        L = int(input_ids.shape[0])
        nc = num_converge_steps if num_converge_steps is not None else self.num_converge_steps
        na = num_attractor_steps if num_attractor_steps is not None else self.num_attractor_steps

        state = torch.zeros(self.state_dim, device=device, dtype=torch.float32)
        attractors = self.dynamics.precompute_attractors(self.embedder, num_steps=na)

        total = torch.zeros((), device=device)
        for t in range(L):
            sig = self.embedder(input_ids[t : t + 1]).squeeze(0)
            state = self.dynamics.converge_fixed(sig, initial_state=state, num_steps=nc)
            logits = self.logits_from_state(state, attractors)
            total = total + F.cross_entropy(logits, target_ids[t : t + 1])
        return total / max(L, 1)

    def evaluate(
        self,
        loader: DataLoader,
        device: torch.device,
    ) -> float:
        """
        Mean cross-entropy loss over ``loader`` (no optimization). Uses the same recurrence
        as :meth:`training_step`.
        """
        self.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for input_ids, target_ids in loader:
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                if input_ids.ndim == 1:
                    loss = self.training_step(input_ids, target_ids)
                else:
                    bsz = input_ids.shape[0]
                    acc = torch.zeros((), device=device)
                    for b in range(bsz):
                        acc = acc + self.training_step(input_ids[b], target_ids[b])
                    loss = acc / max(bsz, 1)
                total += float(loss.item())
                n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 20,
        *,
        num_converge_steps: int | None = None,
        num_attractor_steps: int | None = None,
    ) -> str:
        """Greedy generation from distance logits; decodes with :class:`~attractor_llm.tokenizer.AttractorTokenizer` when set."""
        self.eval()
        device = next(self.parameters()).device
        nc = num_converge_steps if num_converge_steps is not None else self.num_converge_steps
        na = num_attractor_steps if num_attractor_steps is not None else self.num_attractor_steps

        if self.tokenizer is not None:
            ids = self.tokenizer.encode(prompt)
        else:
            words = prompt.split()
            ids = [self._vocab_list.index(w) for w in words if w in self._vocab_list]
        if not ids:
            ids = [0]

        state = torch.zeros(self.state_dim, device=device, dtype=torch.float32)
        attractors = self.dynamics.precompute_attractors(self.embedder, num_steps=na)

        for tid in ids:
            sig = self.embedder(torch.tensor([tid], device=device, dtype=torch.long)).squeeze(0)
            state = self.dynamics.converge_fixed(sig, initial_state=state, num_steps=nc)

        out_ids: list[int] = []
        for _ in range(max_tokens):
            logits = self.logits_from_state(state, attractors)
            nxt = int(logits.argmax(dim=-1).item())
            out_ids.append(nxt)
            sig = self.embedder(torch.tensor([nxt], device=device, dtype=torch.long)).squeeze(0)
            state = self.dynamics.converge_fixed(sig, initial_state=state, num_steps=nc)

        if self.tokenizer is not None:
            return self.tokenizer.decode(out_ids)
        return " ".join(self._vocab_list[i] for i in out_ids if i < len(self._vocab_list))

    def config_dict(self) -> dict[str, object]:
        """Serializable hyperparameters for checkpointing."""
        return {
            "state_dim": self.state_dim,
            "vocab": list(self._vocab_list),
            "vocab_size": self.embedder.vocab_size,
            "num_attractor_steps": self.num_attractor_steps,
            "num_converge_steps": self.num_converge_steps,
            "cubic_scale": float(self.dynamics.cubic_scale.detach().cpu().item()),
            "dt": float(self.dynamics.dt),
            "tokenizer_config": self._tokenizer_config(),
        }

    def _tokenizer_config(self) -> dict[str, object] | None:
        if self.tokenizer is None:
            return None
        tok = self.tokenizer
        return {
            "encoding_name": tok.encoding_name,
            "n_vocab": tok.n_vocab,
            "use_tiktoken": tok.uses_tiktoken,
        }

    def save(self, path: str | Path) -> None:
        """Save weights and hyperparameters (same schema as :func:`attractor_llm.training.save_checkpoint`)."""
        from attractor_llm.training import save_checkpoint

        save_checkpoint(self, path, optimizer=None)

    @classmethod
    def load(cls, path: str | Path, device: torch.device | None = None) -> TorchAttractorLanguageModel:
        """Load from a checkpoint written by :meth:`save` or :func:`attractor_llm.training.save_checkpoint`."""
        from attractor_llm.training import load_checkpoint

        model, _ = load_checkpoint(path, device=device)
        return model


def synthetic_training_demo(
    *,
    state_dim: int = 64,
    steps: int = 50,
    lr: float = 0.05,
    device: torch.device | None = None,
) -> list[float]:
    """
    Tiny synthetic run: random token sequences, minimize :meth:`TorchAttractorLanguageModel.training_step`.
    Returns the loss history (for quick verification that the loss can decrease).
    """
    if device is None:
        device = torch.device("cpu")
    torch.manual_seed(0)
    m = TorchAttractorLanguageModel(
        state_dim=state_dim,
        num_attractor_steps=8,
        num_converge_steps=8,
    ).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    losses: list[float] = []
    vs = m.embedder.vocab_size
    for _ in range(steps):
        L = 4
        x = torch.randint(0, vs, (L,), device=device)
        y = torch.randint(0, vs, (L,), device=device)
        opt.zero_grad()
        loss = m.training_step(x, y)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return losses
