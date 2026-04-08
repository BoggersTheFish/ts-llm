"""
Phase 4 capacity scaling study.

Trains AttractorLM at four sizes and records val_ppl + attractor geometry
metrics (intrinsic_dimensionality, basin_separation_ratio) for each.

Configs
-------
  tiny   fast_dim=64   slow_dim=16   (~560K params at vocab=8192)
  small  fast_dim=128  slow_dim=32   (~2.1M params)
  base   fast_dim=256  slow_dim=64   (~8.4M params)   ← current default
  large  fast_dim=512  slow_dim=128  (~33M params)

Usage
-----
    # Print config table only (no training):
    python scripts/capacity_study.py --dry-run

    # Train a single config:
    python scripts/capacity_study.py --config small --bpe-model data/bpe/wikitext2_8k

    # Train all configs sequentially:
    python scripts/capacity_study.py --all --bpe-model data/bpe/wikitext2_8k

    # Load existing checkpoints and compare geometry (no training):
    python scripts/capacity_study.py --compare --bpe-model data/bpe/wikitext2_8k

Results are saved to data/capacity_study.json after each config completes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from dataclasses import asdict

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from model import AttractorConfig, AttractorLM


# ---------------------------------------------------------------------------
# Config table
# ---------------------------------------------------------------------------

#: Named configs in order of increasing capacity.
STUDY_CONFIGS: list[tuple[str, int, int]] = [
    ("tiny",  64,  16),
    ("small", 128, 32),
    ("base",  256, 64),
    ("large", 512, 128),
]


def count_params(fast_dim: int, slow_dim: int, vocab_size: int) -> int:
    """Build a throwaway model and count parameters."""
    cfg = AttractorConfig(vocab_size=vocab_size, fast_dim=fast_dim, slow_dim=slow_dim)
    model = AttractorLM(cfg)
    return sum(p.numel() for p in model.parameters())


def checkpoint_path(name: str) -> Path:
    return _ROOT / "checkpoints" / f"attractor_{name}.pt"


def detect_vocab(bpe_model: str | None) -> int | None:
    """Return the actual vocab size from the tokenizer, or None if unavailable."""
    if not bpe_model:
        return None
    try:
        from tokenizer import BPETokenizer
        tok = BPETokenizer(bpe_model + ".model")
        return tok.vocab_size()
    except Exception:
        return None


def print_config_table(vocab_size: int = 8192, bpe_model: str | None = None) -> None:
    actual = detect_vocab(bpe_model)
    if actual is not None and actual != vocab_size:
        print(f"  (tokenizer vocab={actual:,}; using that for param counts)")
        vocab_size = actual
    print(f"\n{'Config':<8}  {'fast_dim':>8}  {'slow_dim':>8}  {'params':>10}  {'checkpoint'}")
    print("-" * 72)
    for name, fd, sd in STUDY_CONFIGS:
        n = count_params(fd, sd, vocab_size)
        ckpt = checkpoint_path(name)
        exists = "✓" if ckpt.exists() else "—"
        print(f"{name:<8}  {fd:>8}  {sd:>8}  {n:>10,}  {exists}  {ckpt}")
    print()


# ---------------------------------------------------------------------------
# Single-config training
# ---------------------------------------------------------------------------

def train_config(
    name: str,
    fast_dim: int,
    slow_dim: int,
    bpe_model: str,
    *,
    epochs: int,
    lr: float,
    warmup_steps: int,
    lr_min_ratio: float,
    chunk_size: int,
    grad_accum: int,
    max_grad_norm: float,
    vocab_size: int,
    seed: int,
) -> dict:
    """Train one config and return a results dict."""
    from train import train, perplexity
    from tokenizer import BPETokenizer
    from data_loader import WikiText2

    device = torch.device("cpu")
    torch.manual_seed(seed)

    tok = BPETokenizer(bpe_model + ".model")
    actual_vocab = tok.vocab_size()

    print(f"\n{'=' * 60}")
    print(f"  Config: {name}  fast_dim={fast_dim}  slow_dim={slow_dim}")
    print(f"{'=' * 60}")

    train_ds = WikiText2("train", tok)
    val_ds   = WikiText2("validation", tok)
    _ = len(train_ds); _ = len(val_ds)
    print(f"  train: {len(train_ds)} docs / {train_ds.total_tokens():,} tokens")
    print(f"  val:   {len(val_ds)} docs / {val_ds.total_tokens():,} tokens")

    cfg = AttractorConfig(
        vocab_size=actual_vocab,
        fast_dim=fast_dim,
        slow_dim=slow_dim,
    )
    model = AttractorLM(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")

    ckpt = checkpoint_path(name)
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    train(
        model, train_ds, val_ds, device,
        epochs=epochs,
        lr=lr,
        lr_min_ratio=lr_min_ratio,
        warmup_steps=warmup_steps,
        chunk_size=chunk_size,
        grad_accum_steps=grad_accum,
        max_grad_norm=max_grad_norm,
        checkpoint_path=str(ckpt),
    )
    elapsed = time.time() - t0
    print(f"  training done in {elapsed:.0f}s")

    return {
        "config": name,
        "fast_dim": fast_dim,
        "slow_dim": slow_dim,
        "n_params": n_params,
        "vocab_size": actual_vocab,
        "epochs": epochs,
        "elapsed_s": elapsed,
        "checkpoint": str(ckpt),
    }


# ---------------------------------------------------------------------------
# Probe a trained checkpoint for geometry metrics
# ---------------------------------------------------------------------------

def probe_config(name: str, bpe_model: str, n_docs: int = 50, step_tokens: int = 50, seed: int = 0) -> dict:
    """Run probe_attractors on a saved checkpoint and return metrics."""
    import argparse as _ap
    import importlib.util

    probe_path = _ROOT / "scripts" / "probe_attractors.py"
    spec = importlib.util.spec_from_file_location("probe_attractors", probe_path)
    probe_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe_mod)

    ckpt = checkpoint_path(name)
    if not ckpt.exists():
        return {"error": f"checkpoint not found: {ckpt}"}

    args = _ap.Namespace(
        checkpoint=str(ckpt),
        bpe_model=bpe_model,
        n_docs=n_docs,
        step_tokens=step_tokens,
        max_relax_steps=100,
        relax_tol=1e-4,
        save=False,
        seed=seed,
    )
    return probe_mod._run_wikitext_probe(args)


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(results: list[dict]) -> None:
    """Print a summary table from a list of per-config result dicts."""
    hdr = (
        f"{'config':<8}  {'fast_dim':>8}  {'slow_dim':>8}  {'params':>10}  "
        f"{'val_ppl':>8}  {'idim_fast':>10}  {'idim_slow':>10}  {'bsr_fast':>10}"
    )
    print(f"\n{'=' * len(hdr)}")
    print("CAPACITY STUDY RESULTS")
    print('=' * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        name     = r.get("config", "?")
        fd       = r.get("fast_dim", "?")
        sd       = r.get("slow_dim", "?")
        n_params = r.get("n_params", 0)
        val_ppl  = r.get("val_ppl", float("nan"))
        hf       = r.get("h_fast", {})
        hs       = r.get("h_slow", {})
        idim_f   = hf.get("intrinsic_dimensionality") or float("nan")
        idim_s   = hs.get("intrinsic_dimensionality") or float("nan")
        bsr_f    = hf.get("basin_separation_ratio") or float("nan")
        print(
            f"{name:<8}  {fd:>8}  {sd:>8}  {n_params:>10,}  "
            f"{val_ppl:>8.2f}  {idim_f:>10.2f}  {idim_s:>10.2f}  {bsr_f:>10.4f}"
        )
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4 capacity scaling study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", choices=[n for n, *_ in STUDY_CONFIGS],
                        help="Train a single named config")
    parser.add_argument("--all", action="store_true",
                        help="Train all configs sequentially")
    parser.add_argument("--compare", action="store_true",
                        help="Load existing checkpoints and compare (no training)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config table and exit")
    parser.add_argument("--bpe-model", default="data/bpe/wikitext2_8k")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--lr-min-ratio", type=float, default=0.1)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--n-docs", type=int, default=50,
                        help="Docs to probe per config for geometry metrics")
    parser.add_argument("--step-tokens", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", action="store_true",
                        help="Append results to data/capacity_study.json")
    args = parser.parse_args()

    print_config_table(vocab_size=args.vocab_size, bpe_model=args.bpe_model)

    if args.dry_run:
        return

    out_path = _ROOT / "data" / "capacity_study.json"
    existing: list[dict] = []
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)

    # Which configs to process
    if args.compare:
        to_probe = [n for n, *_ in STUDY_CONFIGS]
        train_them = False
    elif args.all:
        to_probe = [n for n, *_ in STUDY_CONFIGS]
        train_them = True
    elif args.config:
        to_probe = [args.config]
        train_them = True
    else:
        parser.print_help()
        return

    results: list[dict] = list(existing)

    config_map = {n: (fd, sd) for n, fd, sd in STUDY_CONFIGS}

    for name in to_probe:
        fd, sd = config_map[name]

        entry: dict = {"config": name, "fast_dim": fd, "slow_dim": sd}

        if train_them:
            train_result = train_config(
                name, fd, sd, args.bpe_model,
                epochs=args.epochs,
                lr=args.lr,
                warmup_steps=args.warmup_steps,
                lr_min_ratio=args.lr_min_ratio,
                chunk_size=args.chunk_size,
                grad_accum=args.grad_accum,
                max_grad_norm=args.max_grad_norm,
                vocab_size=args.vocab_size,
                seed=args.seed,
            )
            entry.update(train_result)

        print(f"\nProbing geometry for '{name}' …")
        probe_result = probe_config(
            name, args.bpe_model,
            n_docs=args.n_docs,
            step_tokens=args.step_tokens,
            seed=args.seed,
        )
        # Extract val_ppl from checkpoint if not already set from training
        if "val_ppl" not in entry and "val_ppl" in probe_result:
            entry["val_ppl"] = probe_result["val_ppl"]
        entry.update({k: v for k, v in probe_result.items() if k not in entry})

        # Update or append
        results = [r for r in results if r.get("config") != name]
        results.append(entry)

        if args.save:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  → saved to {out_path}")

    print_comparison_table(results)

    if not args.save:
        print("(Run with --save to write results to data/capacity_study.json)")


if __name__ == "__main__":
    main()
