# TinyLLM

TinyLLM is a recurrent attractor-language-model research repo. It models sequence state with iterative relaxation dynamics rather than self-attention.

The codebase has two tracks:

1. **Frozen evaluation harness** (`main.py`, `eval_harness.py`)  
   Word-level `MinimalAttractorLM` on JSON corpora with detailed geometry diagnostics and optional contrastive training.
2. **WikiText-2 training** (`train.py`, `model.py`)  
   BPE tokenizer + two-timescale `AttractorLM` (`h_fast`, `h_slow`) with chunked document training and optional DEQ (`model_deq.py`, `--use-deq`).

Core toy-model update per relaxation substep:

`h <- h + alpha * (tanh(W h + W_x x) - h)`

where `x` is the current token embedding.

## Data Configuration (Harness)

- **[`data/training.json`](data/training.json)**  
  Corpus, branch oversampling controls (`branch_line_count`), next-token checks (`branch_tests`), optional ambiguity diagnostics, optional gate thresholds, and optional contrastive settings.
- **[`data/prompts.json`](data/prompts.json)**  
  Prompt-based generation settings (`greedy_prefixes`, `greedy_max_len`, optional temperature/top-k, loop/cursor generation options).

To expand the harness corpus, use [`scripts/generate_training_corpus.py`](scripts/generate_training_corpus.py), or set `TINYLLM_TRAINING_DATA` to a custom JSON path.

### Generation Modes (`data/prompts.json`)

Priority order: `cursor_generation.enabled` -> `loop_prevention.enabled` -> plain `generate`.

- **Plain**: `generate()` (length-bounded only)
- **Loop prevention**: `generate_with_loop_prevention()` (EOS, attractor-cosine threshold, repeated n-gram checks)
- **Cursor generation**: `generate_with_cursor()` (EOS, repetition window, optional return-to-attractor checks)

After decoding, the harness reports generation metrics:
- `exact_match_rate`
- `longest_repeated_ngram`

Environment overrides:
- `TINYLLM_TRAINING_DATA`
- `TINYLLM_PROMPTS_DATA`

Helper functions in [`utils.py`](utils.py) include prefix divergence utilities (`first_divergence_index`, `shared_prefix_until_divergence`).

## Attractor Geometry and Analysis

Key geometry tools:
- [`eval_harness.py`](eval_harness.py): `get_state_for_tokens`, `collect_prefix_states`, `trace_states`, `trace_prefix`
- [`state_analysis.py`](state_analysis.py): `pairwise_cosine`, `pairwise_euclidean`, `basin_separation_ratio`, `intrinsic_dimensionality`
- [`scripts/probe_attractors.py`](scripts/probe_attractors.py): harness probe mode and checkpoint-based WikiText probe mode

`python main.py` runs the frozen harness and prints training diagnostics, attractor geometry tables, decoding outputs, generation metrics, and gate status. Exit code is `1` if configured gates fail.

## WikiText-2 Training (`train.py`)

Trains [`AttractorLM`](model.py) on WikiText-2 with BPE. Included tokenizer assets:
- [`data/bpe/wikitext2_8k.model`](data/bpe/wikitext2_8k.model)
- [`data/bpe/wikitext2_8k.vocab`](data/bpe/wikitext2_8k.vocab)

```bash
pip install -r requirements-wikitext.txt
python train.py --help
python train.py --bpe-model data/bpe/wikitext2_8k   # typical: uses existing BPE
python train.py --use-deq                           # DEQ fast relax
```

`train.py` includes linear warmup + cosine LR decay, gradient norm logging, and spectral-radius monitoring for fast-relax contractivity.

Additional tools:
- [`scripts/capacity_study.py`](scripts/capacity_study.py): run named model-size sweeps (`tiny`, `small`, `base`, `large`)
- [`scripts/probe_attractors.py`](scripts/probe_attractors.py): collect geometry metrics from saved checkpoints

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

### Test Suite

```bash
pip install -r requirements-dev.txt
pytest tests/
```

The full-training regression is marked `slow` (~tens of seconds on CPU). Fast-only:

```bash
pytest tests/ -m "not slow"
```

Targeted examples:

```bash
pytest tests/test_eval_harness.py -m "not slow"
pytest tests/test_model.py tests/test_deq.py
pytest tests/test_capacity_study.py tests/test_state_analysis.py tests/test_train_utils.py
```

## Verified Run (Documented Transcript)

The canonical full transcript (command, complete output, and exit code) is in [`docs/verified_run.md`](docs/verified_run.md).

## License

[MIT](LICENSE).
