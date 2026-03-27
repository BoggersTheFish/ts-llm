#!/usr/bin/env python3
"""Command-line interface for legacy and torch attractor workflows.

Note:
    CLI behavior orchestrates runtime configuration and IO. It does not alter
    the underlying attractor-state update equations.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from attractor_llm import AttractorLanguageModel, GenerationResult
from attractor_llm.model import GenerationConfig


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    """Formatter combining default-value help and raw multiline examples."""


def _batch_token_count(batch_ids: object) -> int:
    """Count token IDs in a potentially nested batch payload.

    Args:
        batch_ids: Batch token payload (tensor/list/list-of-tensors).

    Returns:
        Total number of token IDs represented by the payload.
    """
    import torch

    if isinstance(batch_ids, torch.Tensor):
        return int(batch_ids.numel())
    if isinstance(batch_ids, list):
        if not batch_ids:
            return 0
        first = batch_ids[0]
        if isinstance(first, torch.Tensor):
            return sum(int(t.numel()) for t in batch_ids)
        if isinstance(first, list):
            return sum(len(row) for row in batch_ids)
        return len(batch_ids)
    return 0


def _sparkline(values: list[float], width: int = 48) -> str:
    """Create a compact unicode sparkline from scalar values.

    Args:
        values: Numeric values to encode.
        width: Maximum number of points to display.

    Returns:
        Unicode sparkline string.
    """
    if not values:
        return ""
    bars = "▁▂▃▄▅▆▇█"
    trimmed = values[-width:]
    lo = min(trimmed)
    hi = max(trimmed)
    if hi <= lo:
        return bars[0] * len(trimmed)
    scale = (len(bars) - 1) / (hi - lo)
    return "".join(bars[int((v - lo) * scale)] for v in trimmed)


def _save_loss_plot(train_losses: list[float], val_losses: list[float], out_dir: Path) -> Path | None:
    """Save train/val loss plot if matplotlib is available.

    Args:
        train_losses: Per-epoch training losses.
        val_losses: Per-eval validation losses.
        out_dir: Directory to store output image.

    Returns:
        Saved image path or ``None`` if plotting is unavailable.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed; skipping --plot-loss image output.", file=sys.stderr)
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"loss_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_train = list(range(1, len(train_losses) + 1))
    x_val = list(range(1, len(val_losses) + 1))
    ax.plot(x_train, train_losses, label="train_loss", linewidth=2)
    ax.plot(x_val, val_losses, label="val_loss", linewidth=2)
    ax.set_xlabel("Epoch index")
    ax.set_ylabel("Loss")
    ax.set_title("Training / Validation Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _run_legacy_generate(args: argparse.Namespace) -> None:
    """Run generation using the legacy NumPy attractor model.

    Args:
        args: Parsed CLI arguments.

    Returns:
        None.
    """
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
    """Run torch training mode.

    Args:
        args: Parsed CLI arguments.

    Returns:
        None.

    Note:
        Training orchestration here must preserve attractor math in model modules.
    """
    import random
    import numpy as np
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from attractor_llm.tokenizer import AttractorTokenizer
    from attractor_llm.torch_model import TorchAttractorLanguageModel
    from attractor_llm.training import (
        TextDataset,
        TrainBatchMetrics,
        load_checkpoint,
        save_checkpoint,
        train_epoch,
    )
    from attractor_llm.phase3 import Phase3Adapter, Phase3Controller, Phase3RuntimeState
    from attractor_llm.phase3.self_improve import SelfImproveAdvisor
    from attractor_llm.phase3.config import SelfImproveConfig

    logger = logging.getLogger("ts_llm.train")
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _seed_worker(worker_id: int) -> None:
        worker_seed = int(args.seed) + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32 - 1))
        torch.manual_seed(worker_seed)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

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
            max_tokens=args.tinystories_max_tokens,
            max_windows=args.tinystories_max_windows,
        )
        val_ds = TinyStoriesDataset(
            split="val",
            val_split=args.val_split,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            max_files=args.tinystories_max_files,
            max_tokens=args.tinystories_max_tokens,
            max_windows=args.tinystories_max_windows,
        )
        if len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                worker_init_fn=_seed_worker,
                generator=loader_generator,
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
                worker_init_fn=_seed_worker,
                generator=loader_generator,
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
                worker_init_fn=_seed_worker,
                generator=loader_generator,
            )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=loader_generator,
    )

    print(
        f"Train: {len(train_ds)} windows | {len(train_loader)} batches/epoch | "
        f"{args.epochs} epoch(s) | device={device} | seed={args.seed}",
        flush=True,
    )
    logger.info("Resolved training config: %s", json.dumps(vars(args), sort_keys=True, default=str))

    model: TorchAttractorLanguageModel
    optimizer: optim.Optimizer
    if args.dynamics == "multihead" and args.state_dim % args.heads != 0:
        print(
            "Error: --state-dim must be divisible by --heads for multi-head dynamics.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.hierarchy_levels > 1:
        if args.dynamics != "multihead":
            print("Error: --hierarchy-levels > 1 requires --dynamics multihead.", file=sys.stderr)
            sys.exit(1)
        if args.state_dim % 2 != 0:
            print("Error: --state-dim must be even for hierarchical (fast/slow) dynamics.", file=sys.stderr)
            sys.exit(1)
        if args.heads % 2 != 0:
            print("Error: --heads must be even for multi-timescale dynamics.", file=sys.stderr)
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
            hierarchy_levels=args.hierarchy_levels,
            timescale_ratio=args.timescale_ratio,
            phrase_vocab_size=args.phrase_vocab_size,
            phrase_span=args.phrase_span,
            phrase_attractors=not args.no_phrase_attractors,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_metadata = {"seed": int(args.seed), "config": vars(args).copy()}
    train_losses: list[float] = []
    val_losses: list[float] = []
    phase3_controller = Phase3Controller(enabled=bool(args.phase3), budget_steps=max(args.phase3_budget_steps, 0))
    phase3_adapter = Phase3Adapter()
    phase3_runtime = Phase3RuntimeState(
        clip_scale=1.0,
        constraints_enabled=bool(args.phase3_constraints),
    )
    phase3_advisor = SelfImproveAdvisor(
        SelfImproveConfig(
            enabled=bool(args.phase3 and args.phase3_self_improve),
            warmup_batches=max(args.phase3_warmup_batches, 1),
            window=max(args.phase3_window, 2),
            strength=max(args.phase3_strength, 0.0),
        )
    )
    phase3_start_s = perf_counter()
    phase3_trace_path = out_dir / "phase3_trace.jsonl"
    if args.phase3:
        phase3_trace_path.write_text("", encoding="utf-8")

    if args.phase3 and args.phase3_constraints:
        phase3_adapter.apply(
            {"action": "set_constraints", "params": {"enabled": True}, "reason": "boot", "ttl_steps": 1},
            optimizer=optimizer,
            runtime=phase3_runtime,
        )

    def _current_clip() -> float | None:
        if args.grad_clip <= 0:
            return None
        return float(args.grad_clip) * float(phase3_runtime.clip_scale)

    def _on_batch_end(metrics: TrainBatchMetrics) -> None:
        if not args.phase3:
            return
        elapsed = perf_counter() - phase3_start_s
        if args.phase3_budget_seconds > 0 and elapsed >= float(args.phase3_budget_seconds):
            return
        snapshot = {
            "epoch": int(metrics.epoch),
            "step": int(metrics.step),
            "train_loss": float(metrics.train_loss),
            "val_loss": None,
            "steps_per_sec": float(metrics.steps_per_sec),
            "grad_norm": metrics.grad_norm,
            "timestamp_s": float(metrics.timestamp_s),
        }
        with phase3_trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
        decision = phase3_controller.decide(snapshot)
        apply_result = phase3_adapter.apply(decision, optimizer=optimizer, runtime=phase3_runtime)

        advice = phase3_advisor.observe(float(metrics.train_loss))
        if advice.active:
            phase3_adapter.apply(
                {
                    "action": "adjust_lr",
                    "params": {"lr_scale": float(advice.lr_scale)},
                    "reason": "self_improve_lr",
                    "ttl_steps": 1,
                },
                optimizer=optimizer,
                runtime=phase3_runtime,
            )
            phase3_adapter.apply(
                {
                    "action": "adjust_clip",
                    "params": {"clip_scale": float(advice.clip_scale)},
                    "reason": "self_improve_clip",
                    "ttl_steps": 1,
                },
                optimizer=optimizer,
                runtime=phase3_runtime,
            )

        if args.log_every_batches > 0 and metrics.step % args.log_every_batches == 0:
            logger.info(
                "[phase3] epoch=%s step=%s action=%s msg=%s lr=%.6g clip_scale=%.4f",
                metrics.epoch,
                metrics.step,
                decision["action"],
                apply_result.message,
                float(optimizer.param_groups[0]["lr"]),
                float(phase3_runtime.clip_scale),
            )

    for epoch in range(args.epochs):
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            max_grad_norm_fn=_current_clip,
            progress=args.train_progress,
            epoch=epoch,
            total_epochs=args.epochs,
            log_every_batches=args.log_every_batches,
            on_batch_end=_on_batch_end,
        )
        train_losses.append(float(loss))
        print(f"Epoch {epoch + 1:02d} | train_loss: {loss:.4f}")
        if args.eval_every and (epoch + 1) % args.eval_every == 0 and val_loader is not None:
            val_loss = model.evaluate(val_loader, device)
            val_losses.append(float(val_loss))
            ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
            print(f"  val_loss: {val_loss:.4f}  val_perplexity: {ppl:.2f}")
            if args.plot_loss:
                plot_path = _save_loss_plot(train_losses, val_losses, out_dir)
                print(f"  train_spark: {_sparkline(train_losses)}")
                print(f"  val_spark:   {_sparkline(val_losses)}")
                if plot_path is not None:
                    print(f"  loss_plot: {plot_path}")
        if (epoch + 1) % max(args.save_every, 1) == 0:
            ckpt_path = out_dir / f"attractor_llm_epoch_{epoch + 1}.pt"
            save_checkpoint(model, ckpt_path, optimizer, metadata=checkpoint_metadata)

    final_path = out_dir / "attractor_llm_final.pt"
    save_checkpoint(model, final_path, optimizer, metadata=checkpoint_metadata)
    print("Training complete.")


def _run_benchmark(args: argparse.Namespace) -> None:
    """Run fixed-budget benchmark comparing two state dimensions.

    Args:
        args: Parsed CLI arguments.

    Returns:
        None.
    """
    import random
    import numpy as np
    from itertools import cycle
    from time import perf_counter

    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from attractor_llm.tokenizer import AttractorTokenizer
    from attractor_llm.torch_model import TorchAttractorLanguageModel
    from attractor_llm.training import TextDataset

    logger = logging.getLogger("ts_llm.benchmark")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    def _seed_worker(worker_id: int) -> None:
        worker_seed = int(args.seed) + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32 - 1))
        torch.manual_seed(worker_seed)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    tokenizer = AttractorTokenizer(
        encoding_name=args.encoding,
        vocab_cap=args.vocab_cap,
        use_tiktoken=not args.no_tiktoken,
    )

    if args.dataset == "tinystories":
        from attractor_llm.datasets import TinyStoriesDataset

        train_ds = TinyStoriesDataset(
            split="train",
            val_split=max(args.val_split, 0.1),
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            max_files=args.tinystories_max_files,
            max_tokens=args.tinystories_max_tokens,
            max_windows=args.tinystories_max_windows,
        )
    else:
        data_path = Path(args.data_file)
        train_ds = TextDataset(
            data_path if data_path.exists() else None,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            split="train",
            val_split=max(args.val_split, 0.1),
        )

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=loader_generator,
    )
    if len(loader) == 0:
        print("Error: benchmark loader is empty; adjust dataset/seq-len settings.", file=sys.stderr)
        sys.exit(1)

    steps = max(args.benchmark_steps, 1)
    state_dims = [args.state_dim, args.benchmark_alt_state_dim]
    results: list[tuple[int, float, float]] = []

    print(
        f"Benchmark: {steps} update steps per config | dataset={args.dataset} | device={device}",
        flush=True,
    )
    logger.info("Resolved benchmark config: %s", json.dumps(vars(args), sort_keys=True, default=str))

    for dim in state_dims:
        if args.dynamics == "multihead" and dim % args.heads != 0:
            print(
                f"Skipping state_dim={dim}: not divisible by heads={args.heads} for multihead.",
                flush=True,
            )
            continue
        model = TorchAttractorLanguageModel(
            state_dim=dim,
            tokenizer=tokenizer,
            dynamics_type=args.dynamics,
            num_heads=args.heads,
            rank=args.rank,
            coupling=args.coupling,
            ode_solver=args.ode_solver or "euler",
            adaptive_ode=args.adaptive,
            ode_atol=args.ode_atol,
            ode_rtol=args.ode_rtol,
            hierarchy_levels=args.hierarchy_levels,
            timescale_ratio=args.timescale_ratio,
            phrase_vocab_size=args.phrase_vocab_size,
            phrase_span=args.phrase_span,
            phrase_attractors=not args.no_phrase_attractors,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        stream = cycle(loader)
        model.train()
        total_loss = 0.0
        t0 = perf_counter()
        window_start = t0
        window_batches = 0
        window_tokens = 0
        for step_idx in range(1, steps + 1):
            input_ids, target_ids = next(stream)
            optimizer.zero_grad()
            loss = model.training_step(input_ids, target_ids)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item())
            window_batches += 1
            window_tokens += _batch_token_count(input_ids)
            if step_idx % 10 == 0:
                now = perf_counter()
                elapsed_window = max(now - window_start, 1e-9)
                batches_per_sec = window_batches / elapsed_window
                tokens_per_sec = window_tokens / elapsed_window
                print(
                    f"    [bench] state_dim={dim} step={step_idx}/{steps} "
                    f"batches/s={batches_per_sec:.2f} tokens/s={tokens_per_sec:.2f} "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )
                window_start = now
                window_batches = 0
                window_tokens = 0
        elapsed = max(perf_counter() - t0, 1e-9)
        avg_loss = total_loss / steps
        steps_per_sec = steps / elapsed
        results.append((dim, avg_loss, steps_per_sec))
        print(
            f"  state_dim={dim:4d} | avg_loss={avg_loss:.4f} | steps/s={steps_per_sec:.2f}",
            flush=True,
        )

    if len(results) >= 2:
        a, b = results[0], results[1]
        faster = a if a[2] >= b[2] else b
        lower = a if a[1] <= b[1] else b
        print(
            f"Summary: faster={faster[0]}D ({faster[2]:.2f} steps/s), "
            f"lower-loss={lower[0]}D ({lower[1]:.4f})",
            flush=True,
        )


def _run_torch_generate(args: argparse.Namespace) -> None:
    """Run generation from a torch checkpoint.

    Args:
        args: Parsed CLI arguments.

    Returns:
        None.
    """
    import torch

    from attractor_llm.phase3.config import ConstraintConfig
    from attractor_llm.phase3.constraint_graph import DeterministicConstraintGraph
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
    constraint_graph = DeterministicConstraintGraph(
        ConstraintConfig(
            enabled=bool(args.phase3 and args.phase3_constraints),
            max_repeat=max(args.phase3_max_repeat, 1),
            repeat_penalty=max(float(args.phase3_repeat_penalty), 0.0),
        )
    )
    text = model.generate(
        args.prompt,
        max_tokens=args.max_tokens,
        logits_adjust_fn=constraint_graph.adjust_logits if constraint_graph.config.enabled else None,
    )
    print(text)


def _run_phase3_simulation(args: argparse.Namespace) -> None:
    """Run offline Phase 3 policy replay from a JSONL metrics trace file."""
    import json as _json

    from attractor_llm.phase3 import Phase3Adapter, Phase3Controller, run_offline_simulation, snapshots_from_dicts

    trace_path = Path(args.phase3_trace)
    if not trace_path.exists():
        print(f"Error: phase3 trace file not found: {trace_path}", file=sys.stderr)
        sys.exit(1)

    rows: list[dict[str, Any]] = []
    for raw in trace_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        rows.append(_json.loads(line))
    snapshots = snapshots_from_dicts(rows)

    controller = Phase3Controller(enabled=True, budget_steps=max(args.phase3_budget_steps, 0))
    adapter = Phase3Adapter()
    results = run_offline_simulation(
        snapshots,
        base_lr=float(args.phase3_base_lr),
        controller=controller,
        adapter=adapter,
    )
    print(f"Phase3 simulation replayed {len(results)} step(s).")
    for item in results[: min(len(results), 20)]:
        print(
            f"  step={item.step:4d} action={item.action:12s} "
            f"lr={item.lr:.6g} clip_scale={item.clip_scale:.4f} reason={item.reason}"
        )


def main() -> None:
    """Parse CLI arguments and dispatch selected mode.

    Returns:
        None.
    """
    p = argparse.ArgumentParser(
        description="ts-llm – attractor LLM: NumPy toy (default) or PyTorch training / torch inference.",
        epilog=(
            "Examples:\n"
            "  ts-llm \"reason about time\" --legacy\n"
            "  ts-llm --mode train --dataset custom --data-file data/train.txt --seed 42\n"
            "  ts-llm --mode train --dataset custom --eval-every 1 --plot-loss\n"
            "  ts-llm --mode benchmark --state-dim 64 --benchmark-alt-state-dim 96 --benchmark-steps 200\n"
            "  ts-llm --mode train --phase3 --phase3-self-improve --phase3-budget-steps 200\n"
            "  ts-llm --config config.yaml --log-level DEBUG --mode train\n"
        ),
        formatter_class=_HelpFormatter,
    )
    p.add_argument(
        "prompt",
        nargs="?",
        default="reason about time and change",
        help="Input prompt (generate mode)",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config file. If omitted and ./config.yaml exists, it is loaded and overrides CLI args.",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity for structured runtime logs.",
    )
    p.add_argument(
        "--mode",
        choices=["generate", "train", "benchmark", "phase3-sim"],
        default="generate",
        help="generate = default CLI",
    )

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
        "--hierarchy-levels",
        type=int,
        default=1,
        help="1 = single timescale (default); 2 = fast token + slow phrase multi-timescale",
    )
    p.add_argument(
        "--timescale-ratio",
        type=float,
        default=4.0,
        help="Slow vs fast: slow block uses dt/ratio and cubic_scale/ratio",
    )
    p.add_argument(
        "--phrase-vocab-size",
        type=int,
        default=8192,
        help="Learned phrase embedding width (hierarchy level 1)",
    )
    p.add_argument(
        "--phrase-span",
        type=int,
        default=4,
        help="Rolling phrase id uses this many recent tokens (inclusive)",
    )
    p.add_argument(
        "--no-phrase-attractors",
        action="store_true",
        help="Map slow path by token id %% phrase_vocab instead of rolling phrase hash",
    )
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
    p.add_argument("--benchmark-steps", type=int, default=200, help="Benchmark mode: optimizer update steps per config")
    p.add_argument(
        "--benchmark-alt-state-dim",
        type=int,
        default=128,
        help="Benchmark mode: alternate state dimension to compare against --state-dim",
    )
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
        help="Optional cap on TinyStories JSON shard files to load (smaller / faster experiments)",
    )
    p.add_argument(
        "--tinystories-max-tokens",
        type=int,
        default=500_000,
        help="Max tokens to load from TinyStories after encoding (0 = full corpus; default 500k for practical CPU training)",
    )
    p.add_argument(
        "--tinystories-max-windows",
        type=int,
        default=2048,
        help="Max sliding windows per split after train/val partition (0 = unlimited). Default 2048 keeps one epoch bounded on CPU",
    )
    p.add_argument("--seq-len", type=int, default=8, help="Sliding window length")
    p.add_argument("--save-every", type=int, default=2, help="Save checkpoint every N epochs (train)")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--eval-every", type=int, default=0, help="Run evaluation every N epochs (0 = disabled)")
    p.add_argument("--grad-clip", type=float, default=1.0, help="Global L2 grad clip (0 = off)")
    p.add_argument(
        "--plot-loss",
        action="store_true",
        help="After each evaluation, save train/val loss PNG and print console sparklines",
    )
    p.add_argument(
        "--log-every-batches",
        type=int,
        default=0,
        help="Optional periodic throughput/loss log interval in batches (0 = off)",
    )
    p.add_argument(
        "--train-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show per-batch tqdm progress during training (default: on; use --no-train-progress to disable)",
    )

    p.add_argument("--encoding", type=str, default="gpt2", help="tiktoken encoding name")
    p.add_argument("--vocab-cap", type=int, default=8192, help="Max embedding / logits width (tiktoken)")
    p.add_argument("--no-tiktoken", action="store_true", help="Use word-list tokenizer (toy vocab)")

    p.add_argument("--state-size", type=int, default=128, help="(Legacy NumPy) state dimension")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Request deterministic torch algorithms for training reproducibility (can be slower)",
    )
    p.add_argument(
        "--phase3",
        action="store_true",
        help="Enable integrated Phase 3 control loop (off by default).",
    )
    p.add_argument(
        "--phase3-budget-steps",
        type=int,
        default=0,
        help="Hard cap on controller decision steps (0 = unlimited while --phase3 is on).",
    )
    p.add_argument(
        "--phase3-budget-seconds",
        type=float,
        default=0.0,
        help="Hard wall-clock cap for Phase 3 decision making in each run (0 = unlimited).",
    )
    p.add_argument(
        "--phase3-self-improve",
        action="store_true",
        help="Enable detached loss-trend advisor for LR/clip scaling.",
    )
    p.add_argument("--phase3-warmup-batches", type=int, default=64, help="Self-improve warmup before advisory actions.")
    p.add_argument("--phase3-window", type=int, default=32, help="Rolling loss window for self-improve policy.")
    p.add_argument("--phase3-strength", type=float, default=0.10, help="Advisory action strength for LR/clip scaling.")
    p.add_argument(
        "--phase3-constraints",
        action="store_true",
        help="Enable deterministic repeat-penalty constraints during torch generation.",
    )
    p.add_argument("--phase3-max-repeat", type=int, default=3, help="Max consecutive token repeats before penalty.")
    p.add_argument(
        "--phase3-repeat-penalty",
        type=float,
        default=0.35,
        help="Logit penalty applied to excessive repeated token during generation.",
    )
    p.add_argument("--phase3-trace", type=str, default="phase3_trace.jsonl", help="JSONL trace for --mode phase3-sim.")
    p.add_argument("--phase3-base-lr", type=float, default=0.001, help="Base LR used for offline --mode phase3-sim.")
    p.add_argument("--no-reset", action="store_true")
    p.add_argument("--candidates", type=str, default="")
    p.add_argument("--list-scores", action="store_true")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--beam-width", type=int, default=1)
    p.add_argument("--beam-depth", type=int, default=1)
    p.add_argument("--diagnostics", action="store_true")

    args = p.parse_args()
    args = _apply_config_overrides(args)
    _setup_logging(args.log_level)

    if args.mode == "train":
        _run_train(args)
        return
    if args.mode == "benchmark":
        _run_benchmark(args)
        return
    if args.mode == "phase3-sim":
        _run_phase3_simulation(args)
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


def _setup_logging(level_name: str) -> None:
    """Configure process-wide logging.

    Args:
        level_name: Logging level string.
    """
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _load_yaml_config(path: Path) -> dict[str, Any]:
    """Load YAML config with OmegaConf or PyYAML.

    Args:
        path: YAML config path.

    Returns:
        Mapping of config key/value pairs.

    Raises:
        RuntimeError: If no supported YAML loader is installed.
        ValueError: If loaded payload is not a mapping.
    """
    data: Any = None
    try:
        from omegaconf import OmegaConf
    except ImportError:
        OmegaConf = None  # type: ignore[assignment]
    if OmegaConf is not None:
        data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    else:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "Config file loading requires OmegaConf or PyYAML. "
                "Install one of them to use --config/config.yaml overrides."
            ) from e
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must decode to a mapping, got: {type(data)}")
    return dict(data)


def _apply_config_overrides(args: argparse.Namespace) -> argparse.Namespace:
    """Apply config.yaml overrides to parsed CLI args.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Updated args namespace.
    """
    explicit = getattr(args, "config", None)
    config_path = Path(explicit) if explicit else Path("config.yaml")
    if not config_path.exists():
        return args
    cfg = _load_yaml_config(config_path)
    for key, value in cfg.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


if __name__ == "__main__":
    main()
