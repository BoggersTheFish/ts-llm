# ts-llm

A **language reasoning substrate** built from **deterministic nonlinear attractor dynamics**. This is **not** a conventional large language model: there is **no pretraining**, **no learned embeddings**, and **no gradient descent**. Behavior comes from **state vectors**, **proto-concept signals** (deterministic maps from words or phrases to vectors), **nonlinear dynamics** (linear diffusion plus a cubic term on a centered state), and **distance-based selection** among candidate continuations.

Repository: [github.com/BoggersTheFish/ts-llm](https://github.com/BoggersTheFish/ts-llm)

## How it works

1. **Inject** a prompt: text is converted to a fixed **signal vector** (`text_to_signal`).
2. **Converge** the internal state with a discrete update until the step change norm falls below a tolerance (default `1e-4`).
3. **Proto-concepts** (vocabulary items) each have a precomputed **attractor** under the same dynamics with that signal held fixed.
4. **Generation** scores candidates with **negative Euclidean distance** from the current state to each candidate’s attractor, injects the best match, re-converges, and repeats.

Optional behaviors include **beam lookahead**, **weighted simultaneous injection**, **sequential injection chains**, **multi-turn state persistence** (do not reset between calls), and **diagnostics** (per-step scores and distances).

## Requirements

- Python 3.10+
- [NumPy](https://numpy.org/) (see `requirements.txt`)

## Quick start

```bash
git clone https://github.com/BoggersTheFish/ts-llm.git
cd ts-llm
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_attractor_llm.py "reason about time and change" --list-scores --max-tokens 12
```

### CLI options (selection)

| Flag | Meaning |
|------|---------|
| `--state-size` | State dimension (default 128) |
| `--max-tokens` | Number of proto-concept steps to emit |
| `--list-scores` | After prompt injection, print top candidate scores |
| `--candidates` | Comma-separated subset of the vocabulary |
| `--beam-width`, `--beam-depth` | Beam search for multi-step lookahead |
| `--diagnostics` | Per-step scores and distances on stderr |
| `--no-reset` | Keep internal state across invocations (multi-turn) |

## Python API

```python
from attractor_llm import AttractorLanguageModel, GenerationConfig, GenerationResult

cfg = GenerationConfig(
    beam_width=3,
    beam_depth=2,
    target_norm=1.0,
)
m = AttractorLanguageModel(state_size=128, config=cfg)
m.reset_state()
m.inject_and_converge("Explain quantum entanglement")

# Greedy or beam-driven generation with optional diagnostics
out = m.generate(
    "Explain quantum entanglement",
    max_tokens=20,
    return_diagnostics=True,
)
if isinstance(out, GenerationResult):
    print(out.text)
    print(out.scores, out.distances)
```

Core dynamics live in `attractor_llm/core.py`; the model, vocabulary, and generation loop are in `attractor_llm/model.py`.

## Project layout

```
ts-llm/
├── attractor_llm/     # Package: dynamics + model
├── run_attractor_llm.py
├── requirements.txt
├── package.json       # Optional future TypeScript tooling (no TS sources yet)
└── README.md
```

## License

Add a `LICENSE` file if you want a specific terms; the repository is otherwise provided as-is for research and experimentation.
