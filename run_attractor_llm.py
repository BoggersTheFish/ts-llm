#!/usr/bin/env python3
"""CLI: inject prompt → converge → iterate proto-concept selection."""

from __future__ import annotations

import argparse
import sys

from attractor_llm import AttractorLanguageModel, GenerationResult
from attractor_llm.model import GenerationConfig


def main() -> None:
    p = argparse.ArgumentParser(description="Attractor dynamics language layer (demo)")
    p.add_argument("prompt", nargs="?", default="reason about time and change", help="Input prompt")
    p.add_argument("--state-size", type=int, default=128)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-reset", action="store_true", help="Keep state across runs (multi-turn)")
    p.add_argument("--candidates", type=str, default="", help="Comma-separated subset of vocabulary")
    p.add_argument("--list-scores", action="store_true", help="Print top candidate scores after prompt")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--beam-width", type=int, default=1, help="Beam size for lookahead (>1 enables beam)")
    p.add_argument("--beam-depth", type=int, default=1, help="Steps of lookahead (>=1)")
    p.add_argument("--diagnostics", action="store_true", help="Print per-step scores and distances to stderr")
    args = p.parse_args()

    cfg = GenerationConfig(beam_width=args.beam_width, beam_depth=args.beam_depth)
    model = AttractorLanguageModel(state_size=args.state_size, seed=args.seed, config=cfg)
    cand = None
    if args.candidates.strip():
        cand = [w.strip() for w in args.candidates.split(",") if w.strip()]

    if not args.no_reset:
        model.reset_state()
    model.inject_and_converge(args.prompt)

    if args.list_scores:
        ranked = model.score_candidates(candidates=cand)
        print("Top candidates (score = -distance to attractor):")
        for w, sc in ranked[: args.top_k]:
            print(f"  {w:20s}  {sc:.6f}")

    gen = model.generate(
        args.prompt,
        max_tokens=args.max_tokens,
        candidates=cand,
        reset=False,
        inject_prompt=False,
        return_diagnostics=args.diagnostics,
    )
    if isinstance(gen, GenerationResult):
        print(gen.text)
        if gen.scores and gen.distances:
            for i, ((w, sc), (_w2, dist)) in enumerate(zip(gen.scores, gen.distances)):
                print(f"step {i + 1}: {w!r}  score={sc:.6f}  dist={dist:.6f}", file=sys.stderr)
    else:
        print(gen)


if __name__ == "__main__":
    main()
