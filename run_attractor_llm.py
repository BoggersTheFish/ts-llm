#!/usr/bin/env python3
"""CLI: legacy NumPy generation (default) or PyTorch training / torch generation with ``--checkpoint``."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from attractor_llm import AttractorLanguageModel, GenerationResult
from attractor_llm.model import GenerationConfig


def _run_legacy_generate(args: argparse.Namespace) -> None:
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


def _run_train(args: argparse.Namespace) -> None:
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from attractor_llm.tokenizer import AttractorTokenizer
    from attractor_llm.torch_model import TorchAttractorLanguageModel
    from attractor_llm.training import load_checkpoint, save_checkpoint, train_epoch, TextDataset

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    tokenizer = AttractorTokenizer(
        encoding_name=args.encoding,
        vocab_cap=args.vocab_cap,
        use_tiktoken=not args.no_tiktoken,
    )

    val_loader: DataLoader | None = None

    if args.dataset == "tinystories" and args.val_split == 0.0:
        print(
            "Note: --val-split is 0; validation set is empty. Use e.g. --val-split 0.1 for a holdout.",
            file=sys.stderr,
        )

    if args.dataset == "tinystories":
        from attractor_llm.datasets import TinyStoriesDataset

        train_ds = TinyStoriesDataset(
            split="train",
            val_split=args.val_split,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            max_files=args.tinystories_max_files,
        )
        val_ds = TinyStoriesDataset(
            split="val",
            val_split=args.val_split,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            max_files=args.tinystories_max_files,
        )
        if len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
            )
    elif args.val_split > 0.0:
        data_path = Path(args.data_file)
        train_ds = TextDataset(
            data_path if data_path.exists() else None,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            split="train",
            val_split=args.val_split,
        )
        val_ds = TextDataset(
            data_path if data_path.exists() else None,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            split="val",
            val_split=args.val_split,
        )
        if len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
            )
    else:
        data_path = Path(args.data_file)
        train_ds = TextDataset(data_path if data_path.exists() else None, tokenizer=tokenizer, seq_len=args.seq_len)
        eval_path = Path(args.eval_data_file) if args.eval_data_file else None
        if eval_path and eval_path.exists():
            eval_ds = TextDataset(eval_path, tokenizer=tokenizer, seq_len=args.seq_len)
            val_loader = DataLoader(
                eval_ds,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
            )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model: TorchAttractorLanguageModel
    optimizer: optim.Optimizer
    if args.dynamics == "multihead" and args.state_dim % args.heads != 0:
        print(
            "Error: --state-dim must be divisible by --heads for multi-head dynamics.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.checkpoint and Path(args.checkpoint).exists():
        model, ckpt = load_checkpoint(args.checkpoint, device=device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        if ckpt.get("optimizer_state") is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            except (TypeError, ValueError):
                pass
    else:
        ode_solver = args.ode_solver or "euler"
        model = TorchAttractorLanguageModel(
            state_dim=args.state_dim,
            tokenizer=tokenizer,
            dynamics_type=args.dynamics,
            num_heads=args.heads,
            rank=args.rank,
            coupling=args.coupling,
            ode_solver=ode_solver,
            adaptive_ode=args.adaptive,
            ode_atol=args.ode_atol,
            ode_rtol=args.ode_rtol,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            max_grad_norm=args.grad_clip if args.grad_clip > 0 else None,
        )
        print(f"Epoch {epoch + 1:02d} | train_loss: {loss:.4f}")
        if args.eval_every and (epoch + 1) % args.eval_every == 0 and val_loader is not None:
            val_loss = model.evaluate(val_loader, device)
            ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
            print(f"  val_loss: {val_loss:.4f}  val_perplexity: {ppl:.2f}")
        if (epoch + 1) % max(args.save_every, 1) == 0:
            ckpt_path = out_dir / f"attractor_llm_epoch_{epoch + 1}.pt"
            save_checkpoint(model, ckpt_path, optimizer)

    final_path = out_dir / "attractor_llm_final.pt"
    save_checkpoint(model, final_path, optimizer)
    print("Training complete.")


def _run_torch_generate(args: argparse.Namespace) -> None:
    import torch

    from attractor_llm.torch_model import TorchAttractorLanguageModel

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if not args.checkpoint or not Path(args.checkpoint).exists():
        print("Error: --checkpoint path to a saved .pt file is required for torch generation.", file=sys.stderr)
        sys.exit(1)
    model = TorchAttractorLanguageModel.load(args.checkpoint, device=device)
    if args.adaptive:
        model.ode.adaptive_ode = True
    if args.ode_solver is not None:
        model.ode.ode_solver = args.ode_solver
    text = model.generate(args.prompt, max_tokens=args.max_tokens)
    print(text)


def main() -> None:
    p = argparse.ArgumentParser(
        description="ts-llm – attractor LLM: NumPy toy (default) or PyTorch training / torch inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "prompt",
        nargs="?",
        default="reason about time and change",
        help="Input prompt (generate mode)",
    )
    p.add_argument("--mode", choices=["generate", "train"], default="generate", help="generate = default CLI")

    p.add_argument(
        "--legacy",
        action="store_true",
        help="Force original NumPy toy generation (ignores --checkpoint for generation)",
    )
    p.add_argument("--checkpoint", type=str, default=None, help="Torch .pt for train resume or torch generation")
    p.add_argument("--state-dim", type=int, default=512, help="Torch state dimension D (multi-head: divisible by --heads)")
    p.add_argument(
        "--dynamics",
        choices=["multihead", "full"],
        default="multihead",
        help="multihead = low-rank per-head diffusion (Phase 2); full = dense diffusion (Phase 1)",
    )
    p.add_argument("--heads", type=int, default=4, help="Number of hierarchical attractor heads")
    p.add_argument("--rank", type=int, default=64, help="Low-rank factor rank per head")
    p.add_argument("--coupling", type=float, default=0.01, help="Cross-head residual coupling strength")
    p.add_argument(
        "--ode-solver",
        choices=["euler", "rk4", "dopri5"],
        default=None,
        help="Convergence integrator: euler = manual Euler (default); rk4/dopri5 use torchdiffeq",
    )
    p.add_argument(
        "--adaptive",
        action="store_true",
        help="Use adaptive dopri5 (tolerance-driven); for generate, overrides checkpoint",
    )
    p.add_argument("--ode-atol", type=float, default=1e-4, help="Adaptive solver atol (torchdiffeq)")
    p.add_argument("--ode-rtol", type=float, default=1e-4, help="Adaptive solver rtol (torchdiffeq)")
    p.add_argument("--device", type=str, default=None, help="cpu | cuda (default: auto)")
    p.add_argument("--max-tokens", type=int, default=16)

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--data-file", type=str, default="data/train.txt")
    p.add_argument("--eval-data-file", type=str, default="data/eval.txt", help="Optional eval text (defaults to synthetic if missing)")
    p.add_argument(
        "--dataset",
        choices=["custom", "tinystories"],
        default="custom",
        help="custom = --data-file or synthetic; tinystories = download after interactive confirmation",
    )
    p.add_argument(
        "--val-split",
        type=float,
        default=0.0,
        help="Hold out this fraction of tokens for validation (0 = disabled). For custom data, also disables --eval-data-file when >0.",
    )
    p.add_argument(
        "--tinystories-max-files",
        type=int,
        default=None,
        help="Optional cap on TinyStories .txt files to load (smaller / faster experiments)",
    )
    p.add_argument("--seq-len", type=int, default=8, help="Sliding window length")
    p.add_argument("--save-every", type=int, default=2, help="Save checkpoint every N epochs (train)")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--eval-every", type=int, default=0, help="Run evaluation every N epochs (0 = disabled)")
    p.add_argument("--grad-clip", type=float, default=1.0, help="Global L2 grad clip (0 = off)")

    p.add_argument("--encoding", type=str, default="gpt2", help="tiktoken encoding name")
    p.add_argument("--vocab-cap", type=int, default=8192, help="Max embedding / logits width (tiktoken)")
    p.add_argument("--no-tiktoken", action="store_true", help="Use word-list tokenizer (toy vocab)")

    p.add_argument("--state-size", type=int, default=128, help="(Legacy NumPy) state dimension")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-reset", action="store_true")
    p.add_argument("--candidates", type=str, default="")
    p.add_argument("--list-scores", action="store_true")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--beam-width", type=int, default=1)
    p.add_argument("--beam-depth", type=int, default=1)
    p.add_argument("--diagnostics", action="store_true")

    args = p.parse_args()

    if args.mode == "train":
        _run_train(args)
        return

    if args.checkpoint and not Path(args.checkpoint).exists():
        print(f"Error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    if args.legacy and args.checkpoint:
        print("Note: --legacy ignores --checkpoint; using NumPy toy generation.", file=sys.stderr)

    use_torch = args.checkpoint is not None and Path(args.checkpoint).exists()
    if use_torch and not args.legacy:
        _run_torch_generate(args)
        return

    _run_legacy_generate(args)


if __name__ == "__main__":
    main()
