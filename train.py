"""
Training script for AttractorLM on WikiText-2.

Quickstart
----------
    # First run: trains BPE tokenizer then model
    python train.py

    # Subsequent runs (tokenizer already trained):
    python train.py --bpe-model data/bpe/wikitext2_8k

    # DEQ mode:
    python train.py --use-deq

Full options: python train.py --help

Architecture defaults for Sprint 3
-----------------------------------
    fast_dim  = 256,  slow_dim  = 64
    alpha_fast= 0.25, alpha_slow= 0.02
    BPTT chunk = 64 tokens (within-document; h_slow crosses chunk boundaries)
    grad accum = 8 steps → effective batch of ~512 tokens per optimizer step
    lr        = 3e-4, grad clip = 1.0
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model import ALPHA_FAST, ALPHA_SLOW, BPTT_WINDOW, AttractorConfig, AttractorLM


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def perplexity(
    model: AttractorLM,
    dataset,
    device: torch.device,
    chunk_size: int = BPTT_WINDOW,
) -> float:
    """
    Perplexity on a WikiText2 dataset split.

    Processes each document with forward_chunked (same dynamics as training:
    h reset per document, h_slow carries across BPTT windows).
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for doc_ids in dataset.iter_documents(shuffle=False):
            doc_ids = doc_ids.to(device)
            n = doc_ids.size(0) - 1
            if n < 1:
                continue
            loss = model.forward_chunked(doc_ids, chunk_size=chunk_size)
            total_nll += loss.item() * n
            total_tokens += n
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: AttractorLM,
    train_dataset,
    val_dataset,
    device: torch.device,
    *,
    epochs: int = 10,
    lr: float = 3e-4,
    chunk_size: int = 64,
    grad_accum_steps: int = 8,
    max_grad_norm: float = 1.0,
    log_every_steps: int = 200,
    checkpoint_path: str | None = None,
) -> None:
    """
    Train AttractorLM on WikiText-2 with gradient accumulation.

    Documents are processed one at a time; the gradient accumulation
    window is based on update steps, not documents.  h is reset per
    document (inside forward_chunked).
    """
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_ppl = float("inf")
    global_step = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_docs = 0
        accum_loss = torch.tensor(0.0, device=device)
        t0 = time.time()

        opt.zero_grad()

        for i, doc_ids in enumerate(train_dataset.iter_documents(shuffle=True)):
            doc_ids = doc_ids.to(device)
            if doc_ids.size(0) < 2:
                continue

            loss = model.forward_chunked(doc_ids, chunk_size=chunk_size)
            # Scale by 1/accum_steps so the effective loss matches a larger batch
            (loss / grad_accum_steps).backward()
            accum_loss = accum_loss + loss.detach()
            epoch_docs += 1

            if (i + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
                opt.zero_grad()
                global_step += 1
                epoch_loss += float(accum_loss.item()) / grad_accum_steps
                accum_loss = torch.tensor(0.0, device=device)

                if global_step % log_every_steps == 0:
                    elapsed = time.time() - t0
                    avg_loss = epoch_loss / max(1, global_step % (log_every_steps or 1))
                    print(
                        f"epoch {epoch:2d}  step {global_step:6d}  "
                        f"loss {avg_loss:.4f}  elapsed {elapsed:.0f}s"
                    )

        # Flush any remaining accumulated gradients
        if epoch_docs % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            opt.zero_grad()
            global_step += 1

        # Validation
        val_ppl = perplexity(model, val_dataset, device, chunk_size=chunk_size)
        elapsed = time.time() - t0
        print(
            f"epoch {epoch:2d}  val_ppl={val_ppl:.2f}  "
            f"docs={epoch_docs}  elapsed={elapsed:.0f}s"
        )

        if checkpoint_path and val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "val_ppl": val_ppl,
                    "cfg": model.cfg,
                },
                checkpoint_path,
            )
            print(f"  → checkpoint saved (val_ppl={val_ppl:.2f})")

        model.train()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train AttractorLM on WikiText-2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fast-dim", type=int, default=256)
    parser.add_argument("--slow-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/bpe/wikitext2_8k",
        help="Path prefix for SentencePiece model (without .model extension)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/attractor_lm.pt",
        help="Path to save best checkpoint",
    )
    parser.add_argument(
        "--use-deq",
        action="store_true",
        help="Use DEQ implicit differentiation instead of BPTT in fast relax",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    # ------------------------------------------------------------------
    # BPE tokenizer
    # ------------------------------------------------------------------
    from tokenizer import BPETokenizer, train_bpe

    bpe_model_path = args.bpe_model + ".model"
    if not Path(bpe_model_path).exists():
        print(f"BPE model not found at {bpe_model_path!r} — training now …")
        from data_loader import WikiText2

        # Load raw text via datasets to train the tokeniser
        class _RawLoader:
            """Minimal loader that returns raw text without tokenising."""
            def __init__(self) -> None:
                from datasets import load_dataset  # type: ignore[import]
                from data_loader import _parse_wikitext_documents

                ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
                lines = [item["text"] for item in ds]
                self._docs = _parse_wikitext_documents(lines, min_chars=50)

            @property
            def docs(self) -> list[str]:
                return self._docs

        raw = _RawLoader()
        train_bpe(raw.docs, args.bpe_model, vocab_size=args.vocab_size)
        print(f"BPE model written to {bpe_model_path!r}")

    tokenizer = BPETokenizer(bpe_model_path)
    actual_vocab = tokenizer.vocab_size()
    print(f"Tokenizer: vocab_size={actual_vocab}")

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    from data_loader import WikiText2

    print("Loading datasets …")
    train_ds = WikiText2("train", tokenizer)
    val_ds = WikiText2("validation", tokenizer)

    # Trigger encoding now so we see timing before training starts
    _ = len(train_ds)
    _ = len(val_ds)
    print(
        f"  train: {len(train_ds)} docs / {train_ds.total_tokens():,} tokens"
    )
    print(
        f"  val:   {len(val_ds)} docs / {val_ds.total_tokens():,} tokens"
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    cfg = AttractorConfig(
        vocab_size=actual_vocab,
        fast_dim=args.fast_dim,
        slow_dim=args.slow_dim,
        use_deq=args.use_deq,
    )
    model = AttractorLM(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"\nAttractorLM  fast_dim={cfg.fast_dim}  slow_dim={cfg.slow_dim}  "
        f"vocab={cfg.vocab_size}  params={n_params:,}  "
        f"deq={'yes' if cfg.use_deq else 'no'}"
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    print(
        f"\nTraining {args.epochs} epochs  "
        f"chunk={args.chunk_size}  accum={args.grad_accum}  lr={args.lr}\n"
    )
    train(
        model,
        train_ds,
        val_ds,
        device,
        epochs=args.epochs,
        lr=args.lr,
        chunk_size=args.chunk_size,
        grad_accum_steps=args.grad_accum,
        max_grad_norm=args.max_grad_norm,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
