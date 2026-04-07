# Minimal TS-native LLM proof-of-concept

**How to read this model:** it treats **language state as a dynamical system** that **relaxes** toward **attractor basins** associated with plausible **semantic continuations**—not as a stack of layers that **re-attend** to all previous tokens. There is **no self-attention**; each step folds the current token into state and runs local relaxation dynamics, then reads out next-token logits.

**Relaxation update** (per substep, before `.detach()`): token-conditioned damped step `h ← h + α · (tanh(W h + W_x x) − h)` where **`x`** is the current token embedding, **`W_x`** is a learned linear map (`bias=False`), **`RELAX_ALPHA = 0.25`**, repeated **`RELAX_STEPS`** times per token. Extra relaxation in diagnostics uses the **last prefix token’s** embedding as fixed **`x`**.

Small word-level attractor language model in PyTorch: a hidden state relaxes with a few fixed-point-style steps per token (dynamics-only inner steps; no full backprop through every relaxation iteration). This is a toy attractor/dynamics demo, not a production language model.

Training uses **one cross-entropy loss per sentence** (no concatenation across sentences), so next-token targets are not corrupted at sentence boundaries.

## Data files

- **[`data/training.json`](data/training.json)** — corpus, oversampling split (`branch_line_count`), next-token test cases (`branch_tests`), optional `ambiguous_prefix`, optional `gates` (CE / entropy thresholds), optional `contrastive_pairs` and `contrastive` (`lambda`, `margin`) for object-spacing loss in training.
- **[`data/prompts.json`](data/prompts.json)** — `greedy_prefixes`, `greedy_max_len`, optional `greedy_temperature` and `greedy_top_k` (0 temperature keeps argmax greedy).

To grow the corpus to hundreds of lines, run [`scripts/generate_training_corpus.py`](scripts/generate_training_corpus.py) (or point `TINYLLM_TRAINING_DATA` at the result). **Large corpora make training much slower:** each epoch runs `forward` once per training row (`make_training_ids` oversamples the first `branch_line_count` lines, then includes every other line once).

**Loop prevention (generation):** In [`data/prompts.json`](data/prompts.json), set `"loop_prevention": { "enabled": true, ... }` to use [`generate_with_loop_prevention`](eval_harness.py) in the demo instead of plain `generate`. When enabled, each step uses [`MinimalAttractorLM.recurrent_step`](eval_harness.py) (same dynamics as before), then may stop early on: optional `eos_token` (must be in vocab), cosine similarity above `attractor_cos_threshold` to any vector in `known_attractors` (the demo passes converged diagnostic states), or a repeated length-`loop_window` n-gram of **generated** token ids. Verbose lines print why generation stopped.

Optional env overrides: `TINYLLM_TRAINING_DATA`, `TINYLLM_PROMPTS_DATA` (absolute paths to JSON files).

Helpers in [`utils.py`](utils.py): `first_divergence_index`, `shared_prefix_until_divergence` for comparing token sequences when you add branching corpora.

## State inspection (attractor geometry)

After training, [`eval_harness.py`](eval_harness.py) exposes **`MinimalAttractorLM.get_state_for_tokens`** (final relaxed `h` for a token sequence) and **`collect_prefix_states`** (many prefixes → CPU NumPy vectors). **`trace_states`** / **`trace_prefix`** record the full trajectory (`start` → each `token:…` → each `relax:…` substep) as detached CPU tensors. [`state_analysis.py`](state_analysis.py) adds **`pairwise_cosine`** and **`pairwise_euclidean`** (sorted key order) over final prefix states.

Running **`python main.py`** prints, in order: **pairwise cosines** for the five `the …` animal prefixes (forward states); **trajectory norms** for `"the cat chases the"`; **`relax_until_convergence`** for every diagnostic prefix (steps, norm, limit-cycle hint when present); **attractor stability** (Gaussian perturbation of converged noun attractors, recovery cosine and steps); **attractor interpolation** along `the cat`↔`the dog` and `the cat chases`↔`the dog runs` (blended state + last-token embed, then relax); **pairwise cosines** among converged noun attractors and among the five canonical verb phrases; **verb basin** matrix (shared verb, different subjects); **object basin** and **global object basin** matrices; noun vs verb converged cosines; **forward vs converged** cosines for noun prefixes; then **greedy** generations from [`data/prompts.json`](data/prompts.json).

## Dependencies

- Python 3.10+ recommended
- `torch>=2.0`, `numpy` (see `requirements.txt`)

## Setup

```bash
pip install -r requirements.txt
```

On many Linux distributions (PEP 668), the system Python is “externally managed” and `pip install` may be refused. In that case use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

From the repository root:

```bash
python main.py
```

Same as `python eval_harness.py`.

Trains on the corpus, prints **mean corpus CE**, **next-token tests** from `training.json`, optional ambiguous-prefix stats, then **greedy generations** from `prompts.json`. Exit code **1** if gates in `training.json` fail.

### Frozen eval harness

Defaults (seed, architecture, epoch count) live in [`eval_harness.py`](eval_harness.py). Gate thresholds default from `data/training.json` → `gates`.

```bash
pip install -r requirements-dev.txt
pytest tests/test_eval_harness.py
```

The regression test is marked `slow` (~tens of seconds on CPU).

## Verified run (documented transcript)

The canonical **full** copy-paste transcript for the current default animal corpus and diagnostics is in **[`docs/verified_run.md`](docs/verified_run.md)** (command, full terminal output, exit code 0).

## License

[MIT](LICENSE).
