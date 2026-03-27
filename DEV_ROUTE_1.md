# Dev Route 1

Pragmatic path for improving `ts-llm` without destabilizing the core attractor substrate.

## Scope

- Keep core attractor math unchanged:
  - continuous state dynamics
  - diffusion + cubic nonlinearity
  - distance-to-proto-attractor logits
- Focus on reliability, observability, and repeatable quality gains.
- Defer speculative "Phase 3"-style behavior until baseline metrics are stable.

## Baseline Checklist

Before any model change, verify:

1. Training starts and prints progress quickly.
2. Epoch completes and writes checkpoints.
3. Generation from `checkpoints/attractor_llm_final.pt` is non-empty.
4. Resume training from checkpoint works.

## Recommended Training Profiles

### Local low-power (CPU sanity)

```bash
.venv/bin/python run_attractor_llm.py --mode train \
  --dataset custom \
  --data-file data/train.txt \
  --epochs 1 \
  --state-dim 32 \
  --dynamics full \
  --seq-len 8 \
  --batch-size 1 \
  --lr 0.001 \
  --grad-clip 1.0 \
  --no-tiktoken \
  --eval-every 0 \
  --device cpu
```

### TinyStories bounded run (safe)

```bash
.venv/bin/python run_attractor_llm.py --mode train \
  --dataset tinystories \
  --tinystories-max-files 2 \
  --tinystories-max-tokens 500000 \
  --tinystories-max-windows 2048 \
  --val-split 0.1 \
  --epochs 2 \
  --state-dim 64 \
  --dynamics full \
  --seq-len 12 \
  --batch-size 1 \
  --lr 0.0006 \
  --grad-clip 0.5 \
  --no-tiktoken \
  --eval-every 1 \
  --device cpu
```

### Vast.ai GPU profile

```bash
.venv/bin/python run_attractor_llm.py --mode train \
  --dataset tinystories \
  --tinystories-max-files 10 \
  --tinystories-max-tokens 2000000 \
  --tinystories-max-windows 20000 \
  --val-split 0.1 \
  --epochs 3 \
  --state-dim 192 \
  --dynamics full \
  --seq-len 64 \
  --batch-size 1 \
  --lr 0.0008 \
  --grad-clip 1.0 \
  --no-tiktoken \
  --eval-every 1 \
  --device cuda
```

## Evaluation Gate

After each run:

1. Record `train_loss` / `val_loss` per epoch.
2. Generate from a fixed prompt set.
3. Compare repetition rate qualitatively before/after.

Suggested prompts:

- `Once upon a time`
- `In a small village`
- `The scientist looked at the machine`

## Change Discipline

- Make one meaningful change at a time.
- Prefer additive flags over hard behavior changes.
- Keep backward compatibility for checkpoints (`.get(...)` defaults).
- If a change affects training graph/loss flow, add a minimal smoke test run.

## Immediate Next Candidate Work

1. Add optional periodic throughput logging (batches/sec every N batches).
2. Add `--seed` propagation to torch training path for reproducible runs.
3. Add a tiny scripted "benchmark mode" to compare two configs on fixed budget.

### Benchmark mode (implemented)

Use `--mode benchmark` to compare two `state_dim` values on the same data and fixed update budget.
This gives a quick speed-vs-loss tradeoff before committing to long runs.

#### CPU benchmark preset

```bash
.venv/bin/python run_attractor_llm.py --mode benchmark \
  --dataset custom \
  --data-file data/train.txt \
  --seq-len 8 \
  --batch-size 1 \
  --state-dim 32 \
  --benchmark-alt-state-dim 48 \
  --benchmark-steps 200 \
  --dynamics full \
  --lr 0.001 \
  --grad-clip 1.0 \
  --no-tiktoken \
  --device cpu
```

#### Vast.ai GPU benchmark preset

```bash
.venv/bin/python run_attractor_llm.py --mode benchmark \
  --dataset tinystories \
  --tinystories-max-files 4 \
  --tinystories-max-tokens 1000000 \
  --tinystories-max-windows 4000 \
  --val-split 0.1 \
  --seq-len 32 \
  --batch-size 1 \
  --state-dim 128 \
  --benchmark-alt-state-dim 192 \
  --benchmark-steps 400 \
  --dynamics full \
  --lr 0.0008 \
  --grad-clip 1.0 \
  --no-tiktoken \
  --device cuda
```

#### Picking the winner

- If one config is much faster with similar loss, pick the faster config.
- If one config has materially lower loss with acceptable speed, pick lower loss.
- Re-run benchmark with a second seed before locking a long training run.

### Config and logging (implemented)

- Optional `config.yaml` overrides via `--config` (or auto-load `./config.yaml` when present).
- Structured logging via `--log-level` with a resolved config summary at train/benchmark start.
- Optional `--plot-loss` emits train/val sparklines and PNG curve output after eval.

### Developer experience (implemented)

- Package now exports `__version__` in `attractor_llm.__init__`.
- CLI `--help` includes concrete multi-line examples for legacy, train, benchmark, and config-driven workflows.
- Editable install support via `pyproject.toml` and script entrypoint:

```bash
pip install -e .
ts-llm --help
```

## Non-Goals for Route 1

- No speculative self-reflection loops in core training path.
- No invasive architecture rewrites.
- No irreversible format changes to existing checkpoints.

## Phase 3 Path (fully integrated, opt-in)

Route 1 now includes a complete Phase 3 implementation with default-off activation.

- Canonical specification remains: [`PHASE_3_SPEC.md`](PHASE_3_SPEC.md)
- Integrated components:
  - controller+adapter actions at train step boundaries (`--phase3`)
  - detached self-improve advisory policy (`--phase3-self-improve`)
  - deterministic generation constraints (`--phase3-constraints`)
  - offline replay harness (`--mode phase3-sim`, `--phase3-trace`)
- Runtime controls:
  - `--phase3-budget-steps`, `--phase3-budget-seconds`
  - `--phase3-warmup-batches`, `--phase3-window`, `--phase3-strength`
  - `--phase3-max-repeat`, `--phase3-repeat-penalty`
- Enforcement policy:
  - strict application path (no silent fallback on invalid decisions)
  - additive checkpoint schema and unchanged core attractor equations

