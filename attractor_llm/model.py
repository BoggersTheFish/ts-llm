"""Proto-concept library, injections, and iterative generation on attractor dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from attractor_llm.core import (
    converge,
    euclidean_distance,
    make_diffusion_matrix,
    text_to_signal,
)

DEFAULT_VOCAB = [
    "the",
    "a",
    "is",
    "are",
    "was",
    "were",
    "to",
    "of",
    "in",
    "and",
    "or",
    "not",
    "if",
    "then",
    "else",
    "true",
    "false",
    "yes",
    "no",
    "maybe",
    "because",
    "so",
    "but",
    "when",
    "where",
    "what",
    "who",
    "how",
    "why",
    "this",
    "that",
    "here",
    "there",
    "now",
    "later",
    "before",
    "after",
    "same",
    "different",
    "high",
    "low",
    "up",
    "down",
    "left",
    "right",
    "good",
    "bad",
    "new",
    "old",
    "big",
    "small",
    "fast",
    "slow",
    "hot",
    "cold",
    "light",
    "dark",
    "water",
    "fire",
    "earth",
    "air",
    "life",
    "death",
    "time",
    "space",
    "mind",
    "body",
    "word",
    "thought",
    "reason",
    "cause",
    "effect",
    "system",
    "state",
    "change",
    "stable",
    "flow",
    "attractor",
    "signal",
    "pattern",
    "structure",
    "memory",
    "future",
    "past",
    "present",
    "question",
    "answer",
    "problem",
    "solution",
]


@dataclass
class GenerationResult:
    """Text from proto-concept sequence plus optional per-step scores and distances."""

    text: str
    tokens: List[str]
    scores: Optional[List[Tuple[str, float]]] = None
    distances: Optional[List[Tuple[str, float]]] = None


@dataclass
class GenerationConfig:
    dt: float = 0.05
    tol: float = 1e-4
    max_converge_steps: int = 50_000
    injection_scale: float = 1.0
    cubic_scale: float = 0.05
    magnitude_floor: float = 1e-3
    magnitude_ceiling: Optional[float] = 12.0
    target_norm: Optional[float] = 1.0
    beam_width: int = 1
    beam_depth: int = 1


@dataclass
class AttractorLanguageModel:
    """
    Maintains internal dynamical state; proto-concepts are fixed-point attractors
    under constant signal injection, used to score continuations by proximity.
    """

    state_size: int = 128
    vocab: List[str] = field(default_factory=lambda: list(DEFAULT_VOCAB))
    config: GenerationConfig = field(default_factory=GenerationConfig)
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = np.random.Generator(np.random.PCG64(self.seed))
        self._diffusion = make_diffusion_matrix(self.state_size, self._rng)
        self._state = np.zeros(self.state_size, dtype=np.float64)
        self._signals: dict[str, np.ndarray] = {}
        self._attractors: dict[str, np.ndarray] = {}
        self._build_proto_library()

    def _run_converge(
        self,
        applied_signal: np.ndarray,
        initial_state: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, int, float]:
        return converge(
            self._diffusion,
            applied_signal,
            self.state_size,
            dt=self.config.dt,
            tol=self.config.tol,
            max_steps=self.config.max_converge_steps,
            initial_state=initial_state,
            cubic_scale=self.config.cubic_scale,
            magnitude_floor=self.config.magnitude_floor,
            magnitude_ceiling=self.config.magnitude_ceiling,
            target_norm=self.config.target_norm,
        )

    def _build_proto_library(self) -> None:
        for w in self.vocab:
            sig = text_to_signal(w, self.state_size)
            self._signals[w] = sig
            att, _, _ = self._run_converge(sig * self.config.injection_scale, initial_state=None)
            self._attractors[w] = att

    def add_proto_concepts(self, words: Iterable[str]) -> None:
        """Extend vocabulary and precompute attractors for new tokens."""
        for w in words:
            if w in self._signals:
                continue
            self.vocab.append(w)
            sig = text_to_signal(w, self.state_size)
            self._signals[w] = sig
            att, _, _ = self._run_converge(sig * self.config.injection_scale, initial_state=None)
            self._attractors[w] = att

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    def reset_state(self, small_noise: float = 0.0) -> None:
        self._state = np.zeros(self.state_size, dtype=np.float64)
        if small_noise > 0:
            self._state = self._rng.standard_normal(self.state_size) * small_noise

    def signal_for(self, text: str) -> np.ndarray:
        return text_to_signal(text, self.state_size)

    def inject_and_converge(
        self,
        text: str,
        *,
        scale: Optional[float] = None,
        competing: Optional[Sequence[Tuple[str, float]]] = None,
    ) -> np.ndarray:
        """
        Inject prompt (and optional weighted competing proto-labels) and re-converge.
        competing: list of (text, weight) blended into applied_signal.
        """
        s = scale if scale is not None else self.config.injection_scale
        applied = self.signal_for(text) * s
        if competing:
            for phrase, w in competing:
                applied = applied + w * self.signal_for(phrase)
        self._state, _, _ = self._run_converge(applied, initial_state=self._state)
        return self._state.copy()

    def inject_sequence(self, phrases: Sequence[str], *, scale: Optional[float] = None) -> np.ndarray:
        """Sequential chaining: inject each proto-concept in order, re-converging each time."""
        for phrase in phrases:
            self.inject_and_converge(phrase, scale=scale)
        return self._state.copy()

    def inject_simultaneous(
        self,
        weighted: Sequence[Tuple[str, float]],
        *,
        scale: Optional[float] = None,
    ) -> np.ndarray:
        """Blend multiple proto-concepts with weights, then converge (competing / conditional blend)."""
        s = scale if scale is not None else self.config.injection_scale
        applied = np.zeros(self.state_size, dtype=np.float64)
        for phrase, w in weighted:
            applied = applied + w * self.signal_for(phrase) * s
        self._state, _, _ = self._run_converge(applied, initial_state=self._state)
        return self._state.copy()

    def semantic_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return euclidean_distance(a, b)

    def score_candidates(
        self,
        candidates: Optional[Sequence[str]] = None,
        *,
        state: Optional[np.ndarray] = None,
        return_distances: bool = False,
    ) -> Union[List[Tuple[str, float]], List[Tuple[str, float, float]]]:
        """
        score(candidate) = -distance(state, candidate_attractor). Higher is better.
        Use state=... to score from an arbitrary state (e.g. beam lookahead).
        """
        st = self._state if state is None else state
        pool = list(candidates) if candidates is not None else self.vocab
        rows: List[Tuple[str, float, float]] = []
        for w in pool:
            if w not in self._attractors:
                self.add_proto_concepts([w])
            d = self.semantic_distance(st, self._attractors[w])
            rows.append((w, -d, d))
        rows.sort(key=lambda x: x[1], reverse=True)
        if return_distances:
            return rows
        return [(w, sc) for w, sc, _ in rows]

    def score_candidates_from_state(
        self,
        state_vec: np.ndarray,
        candidates: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, float, float]]:
        """Rank proto-concepts from a given state vector (same scoring as score_candidates)."""
        return self.score_candidates(candidates, state=state_vec, return_distances=True)  # type: ignore[return-value]

    def best_continuation(
        self,
        candidates: Optional[Sequence[str]] = None,
    ) -> Tuple[str, float]:
        ranked = self.score_candidates(candidates)
        return ranked[0]

    def beam_next_token(
        self,
        candidates: Optional[Sequence[str]] = None,
    ) -> Tuple[str, float]:
        """
        Beam lookahead: sum path scores over beam_depth steps, keep top beam_width paths.
        First token of the best path is returned (requires beam_depth >= 1).
        """
        pool = list(candidates) if candidates is not None else list(self.vocab)
        bw = max(1, self.config.beam_width)
        bd = max(1, self.config.beam_depth)
        if bw == 1 and bd == 1:
            return self.best_continuation(pool)

        beam: List[Tuple[float, np.ndarray, Optional[str]]] = [(0.0, self._state.copy(), None)]
        for _ in range(bd):
            next_beam: List[Tuple[float, np.ndarray, Optional[str]]] = []
            for path_score, st, first in beam:
                ranked = self.score_candidates_from_state(st, pool)[:bw]
                for w, sc, _dist in ranked:
                    sig = self._signals[w] * self.config.injection_scale
                    new_st, _, _ = self._run_converge(sig, initial_state=st)
                    first_token = first if first is not None else w
                    next_beam.append((path_score + sc, new_st, first_token))
            next_beam.sort(key=lambda x: -x[0])
            beam = next_beam[:bw]
        best = max(beam, key=lambda x: x[0])
        assert best[2] is not None
        return best[2], best[0]

    def step_generation(
        self,
        candidates: Optional[Sequence[str]] = None,
        *,
        inject_selected: bool = True,
        exclude: Optional[Sequence[str]] = None,
        use_beam: Optional[bool] = None,
    ) -> Tuple[str, float]:
        """
        Pick next proto-concept (greedy or beam lookahead), optionally inject and converge.
        Pass exclude to drop recent tokens and reduce immediate repetition in greedy loops.
        """
        pool = list(candidates) if candidates is not None else list(self.vocab)
        if exclude:
            ban = set(exclude)
            pool = [w for w in pool if w not in ban]
        if not pool:
            pool = list(candidates) if candidates is not None else list(self.vocab)
        beam_on = use_beam
        if beam_on is None:
            beam_on = self.config.beam_width > 1 or self.config.beam_depth > 1
        if beam_on:
            word, score = self.beam_next_token(pool)
        else:
            word, score = self.best_continuation(pool)
        if inject_selected:
            self.inject_and_converge(word)
        return word, score

    def generate(
        self,
        prompt: str,
        max_tokens: int = 12,
        candidates: Optional[Sequence[str]] = None,
        *,
        reset: bool = True,
        inject_prompt: bool = True,
        separator: str = " ",
        no_repeat_last: int = 1,
        return_diagnostics: bool = False,
        use_beam: Optional[bool] = None,
    ) -> Union[str, GenerationResult]:
        """
        Full loop: prompt → inject/converge → repeatedly pick nearest attractor → inject.
        Set inject_prompt=False if the state was already converged from this prompt.
        no_repeat_last: exclude the last N generated tokens from the next pick (when possible).
        If return_diagnostics=True, returns GenerationResult with per-step scores and distances.
        """
        if reset:
            self.reset_state()
        if inject_prompt:
            self.inject_and_converge(prompt)
        out: List[str] = []
        step_scores: List[Tuple[str, float]] = []
        step_dists: List[Tuple[str, float]] = []
        pool = list(candidates) if candidates is not None else list(self.vocab)
        for _ in range(max_tokens):
            exclude = out[-no_repeat_last:] if no_repeat_last > 0 and len(out) >= no_repeat_last else None
            pool_step = pool
            if exclude:
                ban = set(exclude)
                pool_step = [w for w in pool if w not in ban] or pool
            snap = self._state.copy() if return_diagnostics else None
            word, score = self.step_generation(
                candidates=pool_step,
                inject_selected=True,
                exclude=None,
                use_beam=use_beam,
            )
            out.append(word)
            step_scores.append((word, score))
            if return_diagnostics and snap is not None:
                dist = self.semantic_distance(snap, self._attractors[word])
                step_dists.append((word, dist))
        text = separator.join(out)
        if return_diagnostics:
            return GenerationResult(text=text, tokens=out, scores=step_scores, distances=step_dists)
        return text
