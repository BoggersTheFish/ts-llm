"""
Attractor geometry probe — Phase 2 instrumentation.

Two operating modes:

  Harness mode (default, no checkpoint required):
      Trains MinimalAttractorLM on the toy corpus, then measures:
        - Pairwise cosine distance matrix between sentence attractors
        - Basin separation ratio (between-class / within-class distance)
        - Intrinsic dimensionality (participation ratio of covariance spectrum)
        - Convergence diagnostics (steps to fixed point, limit-cycle detection)

  WikiText mode (requires a trained checkpoint + BPE model):
      Loads AttractorLM, samples validation documents, probes h_fast and h_slow
      at evenly-spaced token positions, and reports the same geometry metrics
      on the collected states. Saves results to data/probe_results.json.

Usage:
    # Harness probe (self-contained, no external data needed):
    python scripts/probe_attractors.py

    # WikiText probe:
    python scripts/probe_attractors.py \\
        --checkpoint checkpoints/attractor_lm.pt \\
        --bpe-model data/bpe/wikitext2_8k \\
        [--n-docs 50] [--step-tokens 50] [--seed 0]

    # Save results:
    python scripts/probe_attractors.py --save
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on the path when run as a script
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from state_analysis import (
    basin_separation_ratio,
    intrinsic_dimensionality,
    labels_sorted,
    pairwise_cosine,
    pairwise_euclidean,
)


# ---------------------------------------------------------------------------
# Shared printing helpers
# ---------------------------------------------------------------------------

def _print_section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _print_cosine_matrix(states: dict[str, np.ndarray], title: str = "Pairwise cosine distance") -> None:
    keys = labels_sorted(states)
    mat = pairwise_cosine(states)
    labels = [k.split()[-1] if " " in k else k for k in keys]  # last word as label
    w = max(6, max(len(l) for l in labels))
    print(f"\n{title} (1 − cosine_sim):")
    print(" " * (w + 2) + "  ".join(f"{l:>{w}}" for l in labels))
    for i, key in enumerate(keys):
        row = f"{labels[i]:>{w}}"
        for j in range(len(keys)):
            row += f"  {mat[i, j]:>{w}.4f}"
        print(row)


def _print_geometry_summary(
    states: dict[str, np.ndarray],
    class_labels: dict[str, str] | None,
    label: str,
) -> dict:
    keys = labels_sorted(states)
    norms = [float(np.linalg.norm(states[k])) for k in keys]

    bsr = basin_separation_ratio(states, class_labels) if class_labels else float("nan")
    idim = intrinsic_dimensionality(states)

    print(f"\n{label} geometry summary:")
    print(f"  n_states              : {len(keys)}")
    print(f"  mean ||state||        : {np.mean(norms):.4f}")
    print(f"  std  ||state||        : {np.std(norms):.4f}")
    print(f"  intrinsic_dim (PR)    : {idim:.2f}" if idim == idim else "  intrinsic_dim (PR)    : nan")
    if class_labels:
        print(f"  basin_separation_ratio: {bsr:.4f}" if bsr == bsr else "  basin_separation_ratio: nan (all singletons)")
        if bsr == bsr:
            if bsr > 2.0:
                verdict = "strong — classes well separated"
            elif bsr > 1.0:
                verdict = "moderate — some class structure"
            else:
                verdict = "weak — geometry does not encode class structure"
            print(f"  verdict               : {verdict}")

    return {
        "n_states": len(keys),
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms)),
        "intrinsic_dimensionality": float(idim) if idim == idim else None,
        "basin_separation_ratio": float(bsr) if bsr == bsr else None,
    }


# ---------------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------------

def _convergence_stats(steps_list: list[int], limit_cycle_flags: list[bool]) -> dict:
    if not steps_list:
        return {"n_probed": 0}
    return {
        "n_probed": len(steps_list),
        "mean_steps": float(np.mean(steps_list)),
        "max_steps": int(np.max(steps_list)),
        "n_limit_cycles": int(sum(limit_cycle_flags)),
    }


def _print_convergence(stats: dict, label: str = "Convergence") -> None:
    print(f"\n{label} diagnostics:")
    if stats.get("n_probed", 0) == 0:
        print("  (no convergence data)")
        return
    print(f"  n probed      : {stats['n_probed']}")
    print(f"  mean steps    : {stats['mean_steps']:.1f}")
    print(f"  max steps     : {stats['max_steps']}")
    print(f"  limit cycles  : {stats['n_limit_cycles']}")
    if stats["n_limit_cycles"] > 0:
        print("  WARNING: possible limit cycles detected — contractivity may be lost")


# ---------------------------------------------------------------------------
# Harness probe
# ---------------------------------------------------------------------------

def _run_harness_probe(args: argparse.Namespace) -> dict:
    from eval_harness import (
        BRANCH_OVERSAMPLE,
        BRANCH_TESTS,
        CORPUS,
        EVAL_SEED,
        LEARNING_RATE,
        RELAX_STEPS,
        STATE_DIM,
        TRAIN_EPOCHS,
        MinimalAttractorLM,
        make_training_ids,
        train_loop,
    )
    from utils import build_vocab, decode, encode, tokenize

    _print_section("HARNESS PROBE — MinimalAttractorLM on toy corpus")

    torch.manual_seed(EVAL_SEED)
    stoi, itos = build_vocab(CORPUS)
    model = MinimalAttractorLM(
        vocab_size=len(stoi),
        state_dim=STATE_DIM,
        relax_steps=RELAX_STEPS,
    )

    print(f"\nTraining {TRAIN_EPOCHS} epochs … (seed={EVAL_SEED})")
    t0 = time.time()
    data = make_training_ids(stoi)
    train_loop(model, data, TRAIN_EPOCHS, lr=LEARNING_RATE, quiet=True)
    print(f"Done in {time.time() - t0:.1f}s")

    device = torch.device("cpu")
    model.eval()

    # --- Collect forward states (one per corpus sentence) ---
    forward_states: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for sentence in CORPUS:
            ids = encode(tokenize(sentence), stoi)
            t = torch.tensor(ids, dtype=torch.long, device=device)
            h = model.get_state_for_tokens(t)
            forward_states[sentence] = h.numpy()

    # --- Collect converged attractor states (relax_until_convergence) ---
    converged_states: dict[str, np.ndarray] = {}
    conv_steps: list[int] = []
    conv_limit_cycles: list[bool] = []
    with torch.no_grad():
        for sentence in CORPUS:
            ids = encode(tokenize(sentence), stoi)
            t = torch.tensor(ids, dtype=torch.long, device=device)
            x_last = model.embed(t[-1])
            h_fwd = torch.tensor(forward_states[sentence], device=device)
            rc = model.relax_until_convergence(
                h_fwd,
                x_embed=x_last,
                max_steps=args.max_relax_steps,
                tol=args.relax_tol,
            )
            converged_states[sentence] = rc.final_state.numpy()
            conv_steps.append(rc.num_steps)
            conv_limit_cycles.append(rc.possible_limit_cycle)

    # Class labels: subject noun (second word in "the cat chases …" → "cat")
    def _subject(sentence: str) -> str:
        words = sentence.split()
        return words[1] if len(words) > 1 else words[0]

    class_labels = {s: _subject(s) for s in CORPUS}

    # --- Print results ---
    _print_cosine_matrix(forward_states, title="Forward states — pairwise cosine distance")
    fwd_metrics = _print_geometry_summary(forward_states, class_labels, "Forward states")

    _print_cosine_matrix(converged_states, title="Converged attractors — pairwise cosine distance")
    conv_metrics = _print_geometry_summary(converged_states, class_labels, "Converged attractors")

    conv_stats = _convergence_stats(conv_steps, conv_limit_cycles)
    _print_convergence(conv_stats, label="Relax-until-convergence")

    # --- Branch test accuracy ---
    correct = 0
    print("\nBranch tests:")
    with torch.no_grad():
        for prefix, expected in BRANCH_TESTS:
            ids = encode(tokenize(prefix), stoi)
            t = torch.tensor(ids, dtype=torch.long, device=device)
            h = torch.zeros(model.state_dim, device=device)
            logits = None
            for tid in t:
                logits, h = model.recurrent_step(h, int(tid.item()))
            pred = itos[int(logits.argmax().item())]
            ok = pred == expected
            correct += int(ok)
            print(f"  [{'ok' if ok else 'FAIL'}] {prefix!r} → {pred!r}  (expected {expected!r})")
    print(f"  {correct}/{len(BRANCH_TESTS)} correct")

    return {
        "mode": "harness",
        "epochs": TRAIN_EPOCHS,
        "state_dim": STATE_DIM,
        "relax_steps": RELAX_STEPS,
        "n_sentences": len(CORPUS),
        "forward_states": fwd_metrics,
        "converged_attractors": conv_metrics,
        "convergence": conv_stats,
        "branch_accuracy": correct / len(BRANCH_TESTS),
    }


# ---------------------------------------------------------------------------
# WikiText probe
# ---------------------------------------------------------------------------

def _fast_relax_convergence(
    h_init: torch.Tensor,
    h_slow: torch.Tensor,
    x: torch.Tensor,
    W_ff,
    W_fs,
    W_x_fast,
    alpha: float,
    max_steps: int,
    tol: float,
) -> tuple[torch.Tensor, int, bool]:
    """
    Run the fast relax loop until convergence; return (h_star, n_steps, possible_limit_cycle).
    All inputs are detached tensors (no_grad context assumed).
    """
    h = h_init.clone()
    deltas: list[float] = []
    for step in range(max_steps):
        target = torch.tanh(W_ff(h) + W_fs(h_slow) + W_x_fast(x))
        h_new = h + alpha * (target - h)
        delta = float(torch.linalg.norm(h_new - h))
        deltas.append(delta)
        h = h_new
        if tol > 0.0 and delta < tol:
            return h, step + 1, False
    # Check for possible limit cycle: alternating sign in deltas
    possible_lc = False
    if len(deltas) >= 5:
        signs = [1 if d > 0 else 0 for d in deltas]
        # delta is always positive (norm), so detect oscillation via comparing
        # consecutive deltas: if they alternately increase/decrease many times
        increases = [deltas[i + 1] > deltas[i] for i in range(len(deltas) - 1)]
        flips = sum(increases[i] != increases[i + 1] for i in range(len(increases) - 1))
        possible_lc = flips >= 4
    return h, max_steps, possible_lc


def _run_wikitext_probe(args: argparse.Namespace) -> dict:
    from model import AttractorConfig, AttractorLM

    _print_section("WIKITEXT PROBE — AttractorLM from checkpoint")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        print(f"ERROR: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg: AttractorConfig = ckpt["cfg"]
    model = AttractorLM(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    device = torch.device("cpu")
    print(f"  epoch={ckpt.get('epoch', '?')}  val_ppl={ckpt.get('val_ppl', float('nan')):.2f}")
    print(f"  fast_dim={cfg.fast_dim}  slow_dim={cfg.slow_dim}  vocab={cfg.vocab_size}")

    bpe_path = args.bpe_model
    if bpe_path is None:
        print("ERROR: --bpe-model required for WikiText probe", file=sys.stderr)
        sys.exit(1)

    try:
        from tokenizer import BPETokenizer
        from data_loader import WikiText2
    except ImportError as e:
        print(f"ERROR: {e}\nInstall requirements-wikitext.txt", file=sys.stderr)
        sys.exit(1)

    print(f"Loading tokenizer: {bpe_path}")
    tok = BPETokenizer(bpe_path + ".model")

    print(f"Loading WikiText-2 validation split …")
    try:
        val_ds = WikiText2("validation", tok)
    except Exception as e:
        print(f"ERROR loading WikiText-2: {e}", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    all_docs = list(val_ds.iter_documents(shuffle=False))
    n_docs = min(args.n_docs, len(all_docs))
    chosen_indices = rng.choice(len(all_docs), size=n_docs, replace=False).tolist()
    print(f"Probing {n_docs} documents (step every {args.step_tokens} tokens) …")

    fast_states: dict[str, np.ndarray] = {}
    slow_states: dict[str, np.ndarray] = {}
    class_labels: dict[str, str] = {}   # key → "early" / "middle" / "late"
    conv_steps: list[int] = []
    conv_limit_cycles: list[bool] = []

    with torch.no_grad():
        for doc_i, idx in enumerate(chosen_indices):
            doc_ids = all_docs[idx]
            T = doc_ids.size(0)
            if T < args.step_tokens:
                continue

            h_fast, h_slow = model.zero_state(device)
            for t in range(T - 1):
                _, h_fast, h_slow = model.step(h_fast, h_slow, doc_ids[t])
                if (t + 1) % args.step_tokens == 0:
                    pos_frac = (t + 1) / T
                    if pos_frac < 0.33:
                        band = "early"
                    elif pos_frac < 0.67:
                        band = "middle"
                    else:
                        band = "late"
                    key = f"d{doc_i}_t{t}"
                    fast_states[key] = h_fast.numpy().copy()
                    slow_states[key] = h_slow.numpy().copy()
                    class_labels[key] = band

                    # Convergence diagnostic: run fast relax from current h_fast
                    x_cur = model.embed(doc_ids[t])
                    h_star, n_steps, lc = _fast_relax_convergence(
                        h_fast, h_slow, x_cur,
                        model.W_ff, model.W_fs, model.W_x_fast,
                        cfg.alpha_fast,
                        max_steps=args.max_relax_steps,
                        tol=args.relax_tol,
                    )
                    conv_steps.append(n_steps)
                    conv_limit_cycles.append(lc)

    if not fast_states:
        print("WARNING: no states collected — documents may be shorter than --step-tokens")
        return {"mode": "wikitext", "error": "no states collected"}

    print(f"Collected {len(fast_states)} state samples across {n_docs} documents")

    # --- Print and summarise ---
    fast_metrics = _print_geometry_summary(fast_states, class_labels, "h_fast states")
    slow_metrics = _print_geometry_summary(slow_states, class_labels, "h_slow states")

    conv_stats = _convergence_stats(conv_steps, conv_limit_cycles)
    _print_convergence(conv_stats, label="Fast relax convergence (from sampled positions)")

    # Position-band norm breakdown
    bands = ["early", "middle", "late"]
    print("\nh_fast mean norm by document position:")
    for band in bands:
        band_keys = [k for k in fast_states if class_labels[k] == band]
        if band_keys:
            norms = [float(np.linalg.norm(fast_states[k])) for k in band_keys]
            print(f"  {band:8s}: n={len(band_keys):3d}  mean={np.mean(norms):.4f}  std={np.std(norms):.4f}")

    print("\nh_slow mean norm by document position:")
    for band in bands:
        band_keys = [k for k in slow_states if class_labels[k] == band]
        if band_keys:
            norms = [float(np.linalg.norm(slow_states[k])) for k in band_keys]
            print(f"  {band:8s}: n={len(band_keys):3d}  mean={np.mean(norms):.4f}  std={np.std(norms):.4f}")

    return {
        "mode": "wikitext",
        "checkpoint": str(checkpoint_path),
        "epoch": ckpt.get("epoch"),
        "val_ppl": ckpt.get("val_ppl"),
        "fast_dim": cfg.fast_dim,
        "slow_dim": cfg.slow_dim,
        "n_docs": n_docs,
        "n_states": len(fast_states),
        "step_tokens": args.step_tokens,
        "h_fast": fast_metrics,
        "h_slow": slow_metrics,
        "convergence": conv_stats,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe attractor geometry of a trained model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to AttractorLM checkpoint (.pt). If omitted, runs the harness probe.",
    )
    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/bpe/wikitext2_8k",
        help="SentencePiece model path prefix (without .model). Required for WikiText probe.",
    )
    parser.add_argument(
        "--n-docs",
        type=int,
        default=50,
        help="Number of WikiText-2 validation documents to probe.",
    )
    parser.add_argument(
        "--step-tokens",
        type=int,
        default=50,
        help="Sample a state every N tokens within each document.",
    )
    parser.add_argument(
        "--max-relax-steps",
        type=int,
        default=100,
        help="Maximum iterations for relax-until-convergence.",
    )
    parser.add_argument(
        "--relax-tol",
        type=float,
        default=1e-4,
        help="Convergence tolerance (||h_new - h|| < tol).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for document sampling.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to data/probe_results.json.",
    )
    args = parser.parse_args()

    if args.checkpoint is not None:
        results = _run_wikitext_probe(args)
    else:
        results = _run_harness_probe(args)

    results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    print()
    if args.save:
        out_path = _ROOT / "data" / "probe_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_path}")
    else:
        print("(Run with --save to write results to data/probe_results.json)")


if __name__ == "__main__":
    main()
