# ts-llm

**Version 0.2.0** — see [CHANGELOG.md](CHANGELOG.md) for release history.

A **language reasoning substrate** built from **nonlinear attractor dynamics**. The repository contains:

1. **Legacy NumPy toy** — deterministic hash signals, fixed diffusion, distance-based “generation” over a small word list (no learning).
2. **Trainable PyTorch attractor LM (Phase 1 MVP)** — learnable embeddings, learnable dynamics, **negative-distance logits** to learned proto-attractors, and **tiktoken** subword tokenization.

Repository: [github.com/BoggersTheFish/ts-llm](https://github.com/BoggersTheFish/ts-llm)

---

## Core dynamics (shared idea)

Let \(\mathbf{s}_t \in \mathbb{R}^D\) be the state, \(A \in \mathbb{R}^{D\times D}\) the diffusion operator, \(\alpha\) the cubic gain, and \(\mathbf{u}\) an injected signal. With \(c(\mathbf{s}) = \mathbf{s} - \bar{\mathbf{s}}\) (centered per vector in batch) one explicit Euler step is

\[
\mathbf{s}_{t+1} = \mathbf{s}_t + \Delta t \left( A \mathbf{s}_t + \alpha\, c(\mathbf{s}_t)^{\odot 3} + \mathbf{u} \right).
\]

**Trainable LM:** \(\mathbf{u}\) comes from a **learnable embedding** of the current token. **Proto-attractors** \(\mathbf{a}_v\) are the fixed-step flows from \(\mathbf{0}\) with each vocabulary signal held fixed. Next-token logits are

\[
\ell_v(\mathbf{s}) = -\frac{\|\mathbf{s} - \mathbf{a}_v\|_2}{\tau},
\]

with learnable temperature \(\tau > 0\). Training minimizes **cross-entropy** of these logits against the next token.

---

## Requirements

- Python 3.10+
- `pip install -r requirements.txt` → NumPy, PyTorch, **tiktoken**, torchdiffeq (optional for future Neural-ODE use)

---

## Quick start (legacy NumPy toy)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_attractor_llm.py "reason about time and change" --list-scores --max-tokens 12
```

Force the original toy path explicitly:

```bash
.venv/bin/python run_attractor_llm.py "reason about time" --legacy
```

---

## Trainable attractor LM (Phase 1 MVP)

### Tokenization

- Default: **tiktoken** GPT-2 BPE (`encoding=gpt2`). Token ids are restricted to \([0, n)\) with **`--vocab-cap`** (default **8192**) so the embedding table size stays manageable; ids \(\ge n\) are skipped during encoding (no giant downloads).
- **`--no-tiktoken`**: fall back to the original **word-list** tokenizer over `DEFAULT_VOCAB` (toy, offline).

### Training (no dataset download required)

```bash
mkdir -p data checkpoints
# optional: put UTF-8 text in data/train.txt
.venv/bin/python run_attractor_llm.py --mode train --epochs 5 --lr 0.01 --eval-every 1 --seq-len 16 --grad-clip 1.0
```

Or:

```bash
.venv/bin/python train.py --epochs 3 --lr 0.005 --no-tiktoken   # word tokenizer only
```

Checkpoints are written under **`checkpoints/`** (see `.gitignore`).

### Generation (PyTorch checkpoint)

```bash
.venv/bin/python run_attractor_llm.py "hello world" --checkpoint checkpoints/attractor_llm_final.pt --max-tokens 20
```

### CLI reference (selection)

| Flag | Role |
|------|------|
| `--mode generate` \| `train` | Default: `generate` |
| `--legacy` | Force **NumPy** toy; ignores `--checkpoint` for generation |
| `--checkpoint` | `.pt` file: torch **generate** or **resume train** |
| `--state-dim` | Torch state dimension \(D\) |
| `--encoding` | tiktoken encoding name (default `gpt2`) |
| `--vocab-cap` | Max token id + 1 for embedding / logits |
| `--no-tiktoken` | Word-list tokenizer |
| `--eval-every` | Run `evaluate()` every *N* epochs (0 = off) |
| `--eval-data-file` | Optional eval text (defaults to synthetic if missing) |
| `--grad-clip` | Global L2 clip (0 disables) |
| `--state-size`, `--beam-*`, `--list-scores`, … | **Legacy NumPy** only |

---

## Python API (PyTorch)

```python
from attractor_llm.tokenizer import AttractorTokenizer
from attractor_llm.torch_model import TorchAttractorLanguageModel

tok = AttractorTokenizer(encoding_name="gpt2", vocab_cap=4096)
m = TorchAttractorLanguageModel(state_dim=256, tokenizer=tok).cuda()
# training_step(input_ids_1d, target_ids_1d), evaluate(loader, device), generate(prompt)
```

Legacy NumPy API remains in `attractor_llm.model.AttractorLanguageModel` and `attractor_llm.core`.

---

## Project layout

```
ts-llm/
├── attractor_llm/
│   ├── core.py           # NumPy dynamics (legacy)
│   ├── torch_core.py     # PyTorch dynamics + batched attractors
│   ├── embeddings.py     # LearnableProtoEmbedder
│   ├── tokenizer.py      # tiktoken + word fallback
│   ├── torch_model.py    # TorchAttractorLanguageModel
│   ├── training.py       # Dataset, checkpoints, train_epoch
│   └── model.py          # NumPy toy + DEFAULT_VOCAB
├── run_attractor_llm.py
├── train.py
├── requirements.txt
├── CHANGELOG.md
├── LICENSE
└── README.md
```

---

## Changelog

Release notes and version history: [CHANGELOG.md](CHANGELOG.md).

## License

This project is licensed under the [MIT License](LICENSE).
