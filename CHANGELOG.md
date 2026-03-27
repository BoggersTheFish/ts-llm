# Changelog

All notable changes to **ts-llm** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- Phase 3: self-improvement loop, constraint graphs.
- Phase 3 policy quality improvements and expanded A/B evaluation.

### Added

- Training mode prints dataset summary (windows, batches/epoch, device) and optional per-batch tqdm progress (`--train-progress` default on; `--no-train-progress` to disable).
- TinyStories: `--tinystories-max-tokens` (default 500k) and `--tinystories-max-windows` (default 2048) so one epoch stays bounded on CPU; shared token cache avoids loading the corpus twice for train+val.
- Phase 3 isolated stubs (`attractor_llm/phase3/contracts.py`, `controller.py`, `adapter.py`) added as spec-first scaffolding; not active in default runtime path.
- CLI help now includes concrete multiline examples for new flags/workflows.
- Added optional config and logging controls: `--config`, `--log-level`, and `--plot-loss`.
- Added `pyproject.toml` with `ts-llm` script entrypoint for editable installs.
- Added package `__version__` export from `attractor_llm.__init__`.
- Integrated opt-in Phase 3 runtime controls for training and generation (`--phase3`, `--phase3-self-improve`, `--phase3-constraints`) with bounded decision budgets.
- Added offline Phase 3 simulation harness (`--mode phase3-sim`) for replaying JSONL metrics traces through controller/adapter logic.
- Added strict adapter behavior for Phase 3 decisions (invalid/unknown actions now fail explicitly instead of fallback).
- Added per-step Phase 3 governor caps to bound LR/clip adjustments when controller and self-improve are both active.
- Added shared `--phase3-trace` path semantics for training trace emission and offline `phase3-sim` replay.
- Added optional synthetic cyclical story dataset (`--dataset synthetic`) for quick no-download training checks with `--synthetic-vocab-size` and `--synthetic-num-sequences`.
- Added torch generation sampling controls: `--sampling {argmax,multinomial}` and `--sample-temperature`.
- Added explicit convergence stabilization controls for torch models: `--state-clip`, `--state-norm-min`, `--state-norm-max`, and `--state-target-norm`.

### Fixed

- Training loop now handles TinyStories/DataLoader batch collation robustly (tensor and list-backed batches) without `.ndim` / `.tolist()` type mismatches.
- `TorchAttractorLanguageModel.training_step` now supports both `(L,)` and `(B, L)` token layouts, including TinyStories' collated `list[tensor(B)]` shape.
- Checkpoint loading is more backward-compatible when older checkpoints are missing newer config fields.
- `TextDataset` no longer returns negative length on very short custom text; tiny corpora are padded to minimum sliding-window size.
- Checkpoint loading now restores tokenizer config when available and safely handles optional numeric config fields stored as `null`.
- `phase3-sim` now validates malformed JSON/trace schema with clear error messages.
- Benchmark mode now honors `--deterministic` for seed-paired reproducibility.
- TinyStories loader now runs without hard dependency on tqdm (graceful fallback).

## [0.5.0] — 2026-03-25

### Added

- **`attractor_llm/hierarchy.py`:** `HierarchicalProtoEmbedder` (token + phrase tables, combined `concat` signal) and `MultiTimescaleMultiHeadDynamics` (fast/slow half-state, each `MultiHeadDynamics` with `dt/ratio` and weaker cubic on the slow block).
- **`TorchAttractorLanguageModel`:** `hierarchy_levels`, `timescale_ratio`, `phrase_vocab_size`, `phrase_span`, `phrase_attractors`; rolling phrase id for slow path; logits still from combined state vs token attractors.
- **CLI:** `--hierarchy-levels`, `--timescale-ratio`, `--phrase-vocab-size`, `--phrase-span`, `--no-phrase-attractors`; validation (even `state_dim` / `heads`, `multihead` only).
- **Checkpoints:** `config_dict` + `load_checkpoint` round-trip for hierarchy fields; infer `hierarchy_levels` from `dynamics.fast.*` weights when missing.

### Changed

- **`torch_core._vector_field_dispatch`:** uses `unified_drift` when present so `torchdiffeq` matches multi-timescale Euler semantics.

## [0.3.0] — 2026-03-25

### Added

- **Phase 2 Step 1 — scaling substrate:** `MultiHeadDynamics` in `torch_core.py` — low-rank diffusion per head (`U`, `V`, diagonal), shared cubic nonlinearity, weak cross-head coupling; batched `converge_fixed` for attractor precomputation.
- **`TorchAttractorLanguageModel`:** `dynamics_type` (`multihead` | `full`), `num_heads`, `rank`, `coupling`; default **multi-head** with `state_dim=512`, `num_heads=4`, `rank=64`.
- **CLI:** `--dynamics`, `--heads`, `--rank`, `--coupling`; `--state-dim` default **512** for train; `--state-dim` divisible by `--heads` when `--dynamics multihead`.
- **Checkpoints:** `save_checkpoint` merges full `config_dict`; `load_checkpoint` infers `dynamics_type` from weights (`dynamics.diffusion` → full) for Phase 1 `.pt` files.

### Changed

- `torch_model` defaults align with Phase 2; Phase 1 dense dynamics available via `--dynamics full`.

## [0.2.0] — 2026-03-25

### Added

- **PyTorch trainable attractor LM (Phase 1 MVP):** `TorchAttractorLanguageModel` with learnable `nn.Embedding` signals, learnable diffusion and cubic coefficient, negative-distance logits to learned proto-attractors, and cross-entropy training.
- **`attractor_llm/torch_core.py`:** Differentiable dynamics (`converge_fixed_steps`, adaptive convergence, batched attractor precomputation for large vocab widths).
- **`attractor_llm/embeddings.py`:** `LearnableProtoEmbedder` with unit-norm projection.
- **`attractor_llm/tokenizer.py`:** `AttractorTokenizer` — **tiktoken** (default GPT-2 BPE) with `vocab_cap`, plus word-list fallback to `DEFAULT_VOCAB`.
- **`attractor_llm/training.py`:** `TextDataset`, sliding windows, `save_checkpoint` / `load_checkpoint`, `train_epoch` with optional gradient clipping.
- **`train.py`:** Thin wrapper for `--mode train`.
- **CLI:** `run_attractor_llm.py` supports `--mode generate|train`, `--checkpoint`, `--legacy`, `--eval-every`, `--eval-data-file`, `--grad-clip`, tokenizer flags (`--encoding`, `--vocab-cap`, `--no-tiktoken`).
- **Documentation:** README overhaul with equations, training/generation guide, and CLI reference.

### Changed

- **Imports:** `attractor_llm/__init__.py` optionally exports torch tokenizer and training helpers when PyTorch is available.

### Fixed

- Legacy NumPy path remains the default for generation when no checkpoint is provided; `--legacy` forces the toy explicitly.

## [0.1.0] — 2026-03-25

### Added

- Initial public release: NumPy attractor toy (`attractor_llm/core.py`, `attractor_llm/model.py`), CLI `run_attractor_llm.py`, deterministic hash signals, proto-concept vocabulary, distance-based generation, beam search and diagnostics options.
- **MIT License** (`LICENSE`).
- Dependencies: `numpy`, `requirements.txt` (later expanded for PyTorch).

Git tags (e.g. `v0.2.0`) can be added on GitHub for release compare links.
