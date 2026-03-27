# Changelog

**Source of truth for the attractor workstream (Phases 0–7).** Release history below was migrated from the repository root on 2026-03-27; the root [`CHANGELOG.md`](../CHANGELOG.md) is a short pointer here.

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

---

## Attractor master plan — phase log

New entries (timestamps, commands, metrics, commits) are appended below.

### 2026-03-27T14:05:00 — Phases 0–7 implementation (developer laptop: **no CPU-heavy training**)

**Phase / sub-task:** Full master plan scaffold with **lightweight** verification. Long synthetic training (Phase 0 sanity), GPU TinyStories runs (Phase 3), and `@pytest.mark.heavy` stress tests are **not** executed by default on low-power machines; exact CLI commands are documented below for HPC/GPU hosts.

**Files created or modified (high level):**

- **Docs:** [`docs/core_math.md`](core_math.md), [`docs/attractor-substrate.md`](attractor-substrate.md), [`docs/preprint.md`](preprint.md); root [`CHANGELOG.md`](../CHANGELOG.md) points here.
- **Hygiene:** [`CONTRIBUTING.md`](../CONTRIBUTING.md), [`CODE_OF_CONDUCT.md`](../CODE_OF_CONDUCT.md), [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).
- **Package:** [`ts_attractor/`](../ts_attractor/) (`numpy_demo.py`, `dynamics.py`, `proto_attractors.py`, `training_loop.py`, `controller.py`, `tsos_bridge.py`, `baselines/hopfield.py`, `README.md`, `figures/trajectory_orbit.gif`).
- **Scripts:** [`scripts/reproduce_sandbox.py`](../scripts/reproduce_sandbox.py), [`scripts/plot_limit_cycle_orbit.py`](../scripts/plot_limit_cycle_orbit.py), [`scripts/gen_toy_checkpoints.py`](../scripts/gen_toy_checkpoints.py), [`scripts/gen_trajectory_gifs.py`](../scripts/gen_trajectory_gifs.py), [`scripts/prepare_tinystories_tokenized.py`](../scripts/prepare_tinystories_tokenized.py), [`scripts/train_attractor.py`](../scripts/train_attractor.py) (dry-run default), [`scripts/eval_orbits.py`](../scripts/eval_orbits.py), [`scripts/eval_harness.py`](../scripts/eval_harness.py), [`scripts/benchmark_table.py`](../scripts/benchmark_table.py), [`scripts/human_eval_prep.py`](../scripts/human_eval_prep.py), [`scripts/qualitative_coherence_numpy.py`](../scripts/qualitative_coherence_numpy.py), [`scripts/tsos_episode_sim.py`](../scripts/tsos_episode_sim.py), [`scripts/transformer_baseline_stub.py`](../scripts/transformer_baseline_stub.py), [`scripts/validate_attractor_targets.py`](../scripts/validate_attractor_targets.py).
- **Configs:** [`configs/attractor_10m.yaml`](../configs/attractor_10m.yaml), [`attractor_50m.yaml`](../configs/attractor_50m.yaml), [`attractor_200m.yaml`](../configs/attractor_200m.yaml).
- **Data artifacts:** [`data/tinystories/processed/`](../data/tinystories/processed/) (`train_tokens.npy`, `val_tokens.npy`), [`data/eval_prompts_50.json`](../data/eval_prompts_50.json).
- **Checkpoints:** [`checkpoints/toy/`](../checkpoints/toy/) (`toy_dim_64.npz`, `toy_dim_512.npz`, `toy_dim_4096.npz` — 4096 uses cheap vector, no full \(D\times D\) diffusion build).
- **Plots:** [`plots/limit_cycle_orbit.png`](../plots/limit_cycle_orbit.png), [`plots/limit_cycle_orbit.html`](../plots/limit_cycle_orbit.html).
- **Tests:** [`tests/test_numpy_demo.py`](../tests/test_numpy_demo.py), [`tests/test_pytorch/`](../tests/test_pytorch/), [`tests/test_heavy_stress.py`](../tests/test_heavy_stress.py) (skipped unless `RUN_HEAVY_TESTS=1`), [`pytest.ini`](../pytest.ini), [`tests/fixtures/reproduce_sandbox_golden*.json`](../tests/fixtures/).
- **Build:** [`pyproject.toml`](../pyproject.toml) — `[project.optional-dependencies].dev`, `[tool.setuptools]`, `ts_attractor` package discovery.

**Key observations / metrics:**

- **Reproducibility:** `python scripts/reproduce_sandbox.py` (default dim 64) and `python scripts/reproduce_sandbox.py --dim 512` match committed golden JSONs (norm trajectory).
- **Tests:** `pytest -m "not heavy and not gpu"` → **66 passed**, **2 deselected** (heavy stress). Full suite: **68** collected.
- **mypy:** `mypy ts_attractor` → **Success** (with `follow_imports = skip` in `pyproject.toml`).
- **Phase 0 CPU sanity (8 epochs, large synthetic):** **not run** on this machine per user request; use e.g.  
  `.venv/bin/python run_attractor_llm.py --mode train --dataset synthetic --state-dim 512 --hierarchy-levels 2 --dynamics multihead --heads 8 --synthetic-num-sequences 2000 --epochs 8 --device cpu ...` on a capable host.
- **Phase 3 GPU training:** `scripts/train_attractor.py --config configs/attractor_10m.yaml` prints equivalent command; add `--execute` on CUDA hardware only.

**Exact commands run (this session, lightweight):**

```text
.venv/bin/pytest -q -m "not heavy and not gpu"   # 66 passed, 2 deselected
.venv/bin/mypy ts_attractor --ignore-missing-imports   # Success
PYTHONPATH=. python scripts/reproduce_sandbox.py
PYTHONPATH=. python scripts/reproduce_sandbox.py --dim 512
PYTHONPATH=. python scripts/gen_toy_checkpoints.py
PYTHONPATH=. python scripts/prepare_tinystories_tokenized.py
PYTHONPATH=. python scripts/plot_limit_cycle_orbit.py
```

**Issues / resolutions:**

- **Laptop CPU constraint:** Deferred long training, 8192-dim default stress, and GPU jobs; documented in this entry and in [`CONTRIBUTING.md`](../CONTRIBUTING.md) (`RUN_HEAVY_TESTS`).
- **mypy + torch:** `follow_imports = skip` keeps checks fast; `numpy_demo` type aliases fixed for mypy.
- **`text_dataloader` inner class:** `Dataset` subclass now stores list in `__init__` (fixes `TypeError`).

**Commit message:** `feat(attractor): scaffold Phases 0–7 (ts_attractor, scripts, docs, tests; no heavy runs)`

### 2026-03-27T16:15:00 — Repo diet: remove huge TinyStories blobs + safe push

**Problem:** `.git` had grown to **~7GB** (tracked `data/tinystories/extracted/*.json` ~140MB each + `TinyStories_all_data.tar.gz` ~1.6GB). `git push` **pack-objects** compressed gigabytes of data and pegged CPU/RAM on a laptop.

**Resolution:**

1. **`git filter-repo`** (in venv): `--invert-paths` for `data/tinystories/extracted` and `data/tinystories/TinyStories_all_data.tar.gz` — removed from **all commits**. After rewrite: `.git` **~1MB**, `size-pack` **~350 KiB**.
2. **`.gitignore`** updated so those paths are never committed again.
3. **`data/tinystories/README.md`** — how to download/extract locally.
4. **`scripts/push_safe.sh`** — `pack.compression=0`, `pack.threads=1` for future pushes.
5. **`CONTRIBUTING.md`** — push section.

**Remote:** `origin` was re-added after `filter-repo` (it strips remotes by default).

**Push:** Because history was rewritten, use **`git push --force-with-lease origin dev_route_1_attractor_v1`** (not a plain `git push` if the branch existed on GitHub with old hashes).

**Commit message:** `fix(repo): strip TinyStories blobs from history, add push_safe.sh`
