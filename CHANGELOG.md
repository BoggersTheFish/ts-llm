# Changelog

All notable changes to **ts-llm** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- Larger-scale training recipes, optional datasets, and further architecture phases (hierarchical attractors, multi-timescale dynamics) as separate milestones.

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
