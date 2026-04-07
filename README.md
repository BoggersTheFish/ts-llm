# Minimal TS-native LLM proof-of-concept

**How to read this model:** it treats **language state as a dynamical system** that **relaxes** toward **attractor basins** associated with plausible **semantic continuations**—not as a stack of layers that **re-attend** to all previous tokens. There is **no self-attention**; each step folds the current token into state and runs local relaxation dynamics, then reads out next-token logits.

This repo has **two tracks**:

1. **Frozen eval harness** (`main.py`) — tiny **word-level** `MinimalAttractorLM` on JSON, heavy **attractor diagnostics**, and optional **contrastive** loss. Use this for experiments and regression tests.
2. **WikiText-2 training** (`train.py`) — **BPE** tokenization, document-boundary batching, and a larger **two-timescale** `AttractorLM` (`model.py`), optionally with **DEQ** (`model_deq.py`, `--use-deq`).

**Relaxation update** (harness / toy model, per substep before `.detach()`): token-conditioned damped step `h ← h + α · (tanh(W h + W_x x) − h)` where **`x`** is the current token embedding, **`W_x`** is a learned linear map (`bias=False`), **`RELAX_ALPHA = 0.25`**, repeated **`RELAX_STEPS`** times per token. Extra relaxation in diagnostics uses the **last prefix token’s** embedding as fixed **`x`**. The WikiText model uses a related multi-step relax with fast/slow state; see `model.py`.

Training in the harness uses **one cross-entropy loss per sentence** (no concatenation across sentences), so next-token targets are not corrupted at sentence boundaries.

## Data files (harness)

- **[`data/training.json`](data/training.json)** — corpus, oversampling split (`branch_line_count`), next-token test cases (`branch_tests`), optional `ambiguous_prefix`, optional `gates` (CE / entropy thresholds), optional `contrastive_pairs` and `contrastive` (`lambda`, `margin`) for object-spacing loss in training. When contrastive pairs are set, the demo training log prints **`ctr`** (mean hinge on cosine similarity between prefix hidden states).
- **[`data/prompts.json`](data/prompts.json)** — `greedy_prefixes`, `greedy_max_len`, optional `greedy_temperature` and `greedy_top_k` (**temperature 0** = argmax; **nonzero temperature** = sampling — the label “Greedy generations” in the log still means “decode from prompts,” not necessarily argmax).

To grow the corpus to hundreds of lines, run [`scripts/generate_training_corpus.py`](scripts/generate_training_corpus.py) (or point `TINYLLM_TRAINING_DATA` at the result). **Large corpora make training much slower:** each epoch runs `forward` once per training row (`make_training_ids` oversamples the first `branch_line_count` lines, then includes every other line once).

### Generation modes (`data/prompts.json`)

Priority: **`cursor_generation.enabled`** → **`loop_prevention.enabled`** → plain **`generate`**.

- **Plain** — `generate()`: no early stopping except length.
- **Loop prevention** (`loop_prevention`) — [`generate_with_loop_prevention`](eval_harness.py): optional `eos_token`, cosine to **precomputed** converged diagnostic attractors above `attractor_cos_threshold`, or repeated length-`loop_window` n-gram in the **generated** suffix. Verbose mode prints stop reason.
- **Cursor generation** (`cursor_generation`) — [`generate_with_cursor`](eval_harness.py): EOS (same `eos_token` as loop prevention), repeated **`k_repeat`** token window in the generated suffix, and/or **return to a prior token-level attractor** (forward state cosine-near a stored [`token_level_attractor`](eval_harness.py) from an earlier prefix). Set **`track_attractors`: false** for a cheaper path that only uses repeat + EOS.

After decoding, the demo prints **generation metrics**: **`exact_match_rate`** (fraction of full decoded strings identical to a training line) and **`longest_repeated_ngram`** (max over prompts, n-grams of length 2–5).

Optional env overrides: `TINYLLM_TRAINING_DATA`, `TINYLLM_PROMPTS_DATA` (absolute paths to JSON files).

Helpers in [`utils.py`](utils.py): `first_divergence_index`, `shared_prefix_until_divergence` for comparing token sequences when you add branching corpora.

## State inspection (attractor geometry)

After training, [`eval_harness.py`](eval_harness.py) exposes **`MinimalAttractorLM.get_state_for_tokens`** (final relaxed `h` for a token sequence) and **`collect_prefix_states`** (many prefixes → CPU NumPy vectors). **`trace_states`** / **`trace_prefix`** record the full trajectory (`start` → each `token:…` → each `relax:…` substep) as detached CPU tensors. [`state_analysis.py`](state_analysis.py) adds **`pairwise_cosine`** and **`pairwise_euclidean`** (sorted key order) over final prefix states.

Running **`python main.py`** prints, in order: **pairwise cosines** for the five `the …` animal prefixes (forward states); **trajectory norms** for `"the cat chases the"`; **`relax_until_convergence`** for every diagnostic prefix (steps, norm, limit-cycle hint when present); **attractor stability** (Gaussian perturbation of converged noun attractors, recovery cosine and steps); **attractor interpolation** along `the cat`↔`the dog` and `the cat chases`↔`the dog runs` (blended state + last-token embed, then relax); **pairwise cosines** among converged noun attractors and among the five canonical verb phrases; **verb basin** matrix (shared verb, different subjects); **object basin** and **global object basin** matrices; noun vs verb converged cosines; **forward vs converged** cosines for noun prefixes; **decoding** from `prompts.json` (mode per config above); **generation metrics**; then gate summary. Exit code **1** if `gates` in `training.json` fail.

## WikiText-2 training (`train.py`)

Trains [`AttractorLM`](model.py) on WikiText-2 with BPE. Committed assets: [`data/bpe/wikitext2_8k.model`](data/bpe/wikitext2_8k.model) and [`data/bpe/wikitext2_8k.vocab`](data/bpe/wikitext2_8k.vocab). First-time use without those files will download WikiText and train BPE (slow).

```bash
pip install -r requirements-wikitext.txt
python train.py --help
python train.py --bpe-model data/bpe/wikitext2_8k   # typical: uses existing BPE
python train.py --use-deq                           # DEQ fast relax
```

See docstring in [`train.py`](train.py) for defaults (dimensions, BPTT chunk, lr, checkpoint path).

## Dependencies

- Python 3.10+ recommended
- **Harness:** `torch>=2.0`, `numpy` — [`requirements.txt`](requirements.txt)
- **WikiText path:** also `datasets`, `sentencepiece` — [`requirements-wikitext.txt`](requirements-wikitext.txt)
- **Tests:** [`requirements-dev.txt`](requirements-dev.txt) (pytest)

## Setup

```bash
pip install -r requirements.txt
```

For WikiText training, additionally:

```bash
pip install -r requirements-wikitext.txt
```

On many Linux distributions (PEP 668), the system Python is “externally managed” and `pip install` may be refused. In that case use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

**Harness demo** (from repository root):

```bash
python main.py
```

Same as `python eval_harness.py`.

### Frozen eval harness

Defaults (seed, architecture, epoch count) live in [`eval_harness.py`](eval_harness.py). Gate thresholds default from `data/training.json` → `gates`.

```bash
pip install -r requirements-dev.txt
pytest tests/
```

The full-training regression is marked `slow` (~tens of seconds on CPU). Fast-only:

```bash
pytest tests/ -m "not slow"
```

## Verified run (documented transcript)

The canonical **full** copy-paste transcript for the current default animal corpus, contrastive training log, diagnostics, sampled decodes, and generation metrics is in **[`docs/verified_run.md`](docs/verified_run.md)** (command, full terminal output, exit code 0).

## License

[MIT](LICENSE).
