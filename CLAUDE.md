# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install runtime dependencies (frozen eval harness)
pip install -r requirements.txt

# For WikiText-2 training (train.py): datasets + SentencePiece
pip install -r requirements-wikitext.txt

# Install dev dependencies (adds pytest)
pip install -r requirements-dev.txt

# Run the harness demo (train + evaluate + diagnostics + decode + generation metrics)
python main.py         # same as: python eval_harness.py

# WikiText-2 AttractorLM training (see train.py --help for checkpoint, DEQ, etc.)
python train.py --bpe-model data/bpe/wikitext2_8k
python train.py --use-deq --bpe-model data/bpe/wikitext2_8k

# Tests: all (skip slow with -m "not slow")
pytest tests/

# Slow full-training regression for harness (~tens of seconds on CPU)
pytest tests/test_eval_harness.py -m slow

# Fast harness unit tests only
pytest tests/test_eval_harness.py -m "not slow"

# Model / DEQ unit tests (no full WikiText train)
pytest tests/test_model.py tests/test_deq.py

# Expand JSON corpus (point TINYLLM_TRAINING_DATA at output if needed)
python scripts/generate_training_corpus.py

# Probe attractor geometry (harness mode)
python scripts/probe_attractors.py

# Probe geometry from a trained checkpoint
python scripts/probe_attractors.py --checkpoint checkpoints/attractor_lm.pt --bpe-model data/bpe/wikitext2_8k

# Capacity scaling study (show config table only)
python scripts/capacity_study.py --dry-run

# Capacity scaling study (all configs)
python scripts/capacity_study.py --all --bpe-model data/bpe/wikitext2_8k --save
```

Environment overrides: `TINYLLM_TRAINING_DATA` and `TINYLLM_PROMPTS_DATA` accept absolute paths to alternate JSON data files.

`python main.py` exits **1** if gate thresholds in `data/training.json` → `gates` are not met.

## Architecture

### Harness track (`eval_harness.py`, `main.py`)

**Toy attractor/dynamics language model** — not transformer-based. No self-attention. A single hidden state vector `h` (shape `[STATE_DIM]`) relaxes per token via a damped fixed-point iteration:

```
h ← h + α · (tanh(W·h + Wₓ·x) − h)
```

where `x` is the current token embedding, `RELAX_ALPHA = 0.25`, repeated `RELAX_STEPS` times. The final `h` after processing a prefix is the forward “attractor” state for that prefix; diagnostics also use **`relax_until_convergence`** and **`token_level_attractor`** (deep relax with last-token `x` fixed).

### WikiText track (`train.py`, `model.py`, `model_deq.py`)

Larger **fast/slow** recurrent relax LM, BPE tokenization (`tokenizer.py`), document streaming (`data_loader.py`), BPTT-style **`forward_chunked`**, optional **DEQ** for the fast relax (`--use-deq`).

### Key files

- **`eval_harness.py`** — `MinimalAttractorLM`, `train_and_evaluate`, `run_demo`, branch/ambiguous metrics, `generate` / `generate_with_loop_prevention` / `generate_with_cursor`, `generation_metrics`, `longest_repeated_ngram_length`, gates. Frozen defaults (`EVAL_SEED`, `STATE_DIM`, `RELAX_STEPS`, `TRAIN_EPOCHS`, etc.) at top.
- **`model.py`** / **`model_deq.py`** — `AttractorLM` for WikiText-2.
- **`train.py`** — WikiText training entrypoint.
- **`data_loader.py`** — `WikiText2` dataset wrapper (HuggingFace `datasets`).
- **`tokenizer.py`** — SentencePiece BPE (`BPETokenizer`, `train_bpe`).
- **`utils.py`** — word-level tokenization for harness (`tokenize`, `build_vocab`, `encode`, `decode`) and prefix helpers.
- **`state_analysis.py`** — NumPy geometry: `pairwise_cosine`, `pairwise_euclidean`, `labels_sorted`, `basin_separation_ratio`, `intrinsic_dimensionality`.
- **`main.py`** — shim to `eval_harness.run_demo`.
- **`data/training.json`** — harness corpus, `branch_tests`, `gates`, optional `contrastive_*`.
- **`data/prompts.json`** — decode prefixes, temperature/top-k, `loop_prevention`, `cursor_generation`.

### Harness training details

- One cross-entropy loss **per sentence** (no concatenation across sentence boundaries).
- Optional contrastive object-spacing loss (`contrastive_lambda`, `contrastive_margin`); training log prints **`ctr`** when active.
- `branch_line_count` lines oversampled (`BRANCH_OVERSAMPLE = 4×`); remaining lines once per epoch.

### Generation modes (`data/prompts.json`)

1. **Plain** — `generate()` only.
2. **Loop prevention** (`loop_prevention.enabled`) — EOS, cosine to **precomputed** diagnostic attractors, repeated `loop_window` n-gram in generated suffix.
3. **Cursor** (`cursor_generation.enabled`) — takes precedence over loop prevention: EOS, repeated `k_repeat` window, optional **prior token-level attractor** return (`track_attractors`).

After each demo decode batch, **`generation_metrics`**: `exact_match_rate` vs full training lines, **`longest_repeated_ngram`**.

### Test structure

- **`tests/test_eval_harness.py`** — fast tests (metrics, cursor smoke) + `@pytest.mark.slow` full `train_and_evaluate` + gates.
- **`tests/test_model.py`**, **`tests/test_deq.py`** — WikiText model pieces.
- **`tests/test_state_analysis.py`**, **`tests/test_capacity_study.py`**, **`tests/test_train_utils.py`** — geometry metrics, study tooling, and training utility checks.
