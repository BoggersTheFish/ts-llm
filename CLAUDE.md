# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install runtime dependencies
pip install -r requirements.txt

# Install dev dependencies (adds pytest)
pip install -r requirements-dev.txt

# Run the demo (train + evaluate + diagnostics)
python main.py         # same as: python eval_harness.py

# Run tests (fast only)
pytest tests/test_eval_harness.py

# Run all tests including the slow full-training regression (~tens of seconds on CPU)
pytest tests/test_eval_harness.py -m slow

# Run a single test
pytest tests/test_eval_harness.py::test_longest_repeated_ngram_length

# Expand corpus (run once; produces a JSON file you can point TINYLLM_TRAINING_DATA at)
python scripts/generate_training_corpus.py
```

Environment overrides: `TINYLLM_TRAINING_DATA` and `TINYLLM_PROMPTS_DATA` accept absolute paths to alternate JSON data files.

Exit code 1 if gate thresholds in `data/training.json` → `gates` are not met.

## Architecture

This is a **toy attractor/dynamics language model** — not transformer-based. There is no self-attention. Instead a single hidden state vector `h` (shape `[STATE_DIM]`) relaxes per token via a damped fixed-point iteration:

```
h ← h + α · (tanh(W·h + Wₓ·x) − h)
```

where `x` is the current token embedding, `RELAX_ALPHA = 0.25`, repeated `RELAX_STEPS` times. The final `h` after processing a prefix is the "attractor state" for that prefix.

### Key files

- **`eval_harness.py`** — everything: `MinimalAttractorLM` (the model), training loop (`train_and_evaluate`), greedy/cursor/loop-prevention generation, evaluation metrics, and the `run_demo` entrypoint. Frozen defaults (`EVAL_SEED`, `STATE_DIM`, `RELAX_STEPS`, `TRAIN_EPOCHS`, etc.) live at the top; change them only when intentionally shifting the baseline.
- **`utils.py`** — word-level tokenization (`tokenize`, `build_vocab`, `encode`, `decode`) and prefix-comparison helpers (`first_divergence_index`, `shared_prefix_until_divergence`).
- **`state_analysis.py`** — NumPy-only geometry over attractor states: `pairwise_cosine`, `pairwise_euclidean`, `labels_sorted`. Operates on `dict[str, np.ndarray]` produced by `collect_prefix_states`.
- **`main.py`** — thin shim; delegates to `eval_harness.py`.
- **`data/training.json`** — corpus, oversampling config (`branch_line_count`), next-token tests (`branch_tests`), gate thresholds (`gates`), optional contrastive-pair loss config.
- **`data/prompts.json`** — greedy decode prefixes, temperature/top-k, loop-prevention and cursor-generation config.

### Training details

- One cross-entropy loss **per sentence** (no concatenation across sentence boundaries, so next-token targets are never corrupted at boundaries).
- Optional contrastive object-spacing loss controlled by `contrastive_lambda` and `contrastive_margin` in `data/training.json`.
- `branch_line_count` lines are oversampled (`BRANCH_OVERSAMPLE = 4×`) during training; remaining lines are included once per epoch.

### Generation modes

Three decode modes are available, all configured from `data/prompts.json`:
1. **Greedy** — plain argmax or temperature+top-k sampling.
2. **Loop prevention** (`loop_prevention.enabled`) — stops on EOS token, cosine similarity to a known attractor above threshold, or a repeated n-gram window in generated tokens.
3. **Cursor generation** (`cursor_generation.enabled`) — sliding repeat-window stop + optional return-to-prior-attractor stop.

### Test structure

`tests/test_eval_harness.py` contains fast unit tests for pure functions and a `@pytest.mark.slow` full training regression that runs `train_and_evaluate` and checks all gate thresholds.
