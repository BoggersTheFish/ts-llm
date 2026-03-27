# Demo Model – Trained on Pure Synthetic Word Data (`dev_route_1_attractor_v1`)

This branch (`demo-synthetic-trained-v1`) is set up to train and demo an attractor language model **using only the built-in synthetic word stream** (`--dataset synthetic`). No TinyStories, no external corpora, and no downloaded datasets.

**Checkpoint files are not committed here by default.** Train on your machine, then add `checkpoints/demo_synthetic_v1/` and update the “Example generation output” section below before publishing.

---

## What this branch adds (so your commands work)

1. **`--name` (train)**  
   Sets the checkpoint filename prefix. Training writes:
   - `checkpoints/<checkpoint-dir>/<name>_epoch_<N>.pt` (when `--save-every` hits)
   - `checkpoints/<checkpoint-dir>/<name>_final.pt`  
   Default prefix is `attractor_llm` if you omit `--name`.

2. **`--synthetic-max-windows` (train, `dataset=synthetic`)**  
   Caps how many sliding windows are used **per split** after the train/val partition.  
   - `0` = unlimited (full stream; can be **very** slow on CPU because `synthetic-num-sequences` builds a long token stream and every offset is a window).  
   - e.g. `512` = use only the first 512 windows per split — **same generator settings** (`--synthetic-vocab-size`, `--synthetic-num-sequences`, etc.), just less data per epoch so CPU training finishes in reasonable time.

---

## Environment

```bash
git clone https://github.com/BoggersTheFish/ts-llm.git
cd ts-llm
git checkout demo-synthetic-trained-v1   # or: git checkout dev_route_1_attractor_v1 && git checkout -b demo-synthetic-trained-v1

python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Training (pure synthetic — target hyperparameters)

**Full specification (as requested for the demo):**

```bash
python run_attractor_llm.py --mode train \
  --dataset synthetic \
  --synthetic-vocab-size 120 \
  --synthetic-num-sequences 2500 \
  --state-dim 64 \
  --hierarchy-levels 2 \
  --dynamics multihead \
  --heads 4 \
  --seq-len 14 \
  --epochs 50 \
  --lr 0.0025 \
  --batch-size 8 \
  --device cpu \
  --checkpoint-dir checkpoints/demo_synthetic_v1 \
  --name demo_synthetic_trained_v1
```

- Use `--device cuda` instead of `--device cpu` if you have a suitable GPU.

**Optional CPU-friendly add-on (same run, bounded windows per epoch):**

```bash
  --synthetic-max-windows 512
```

(Add that line to the command above.) This keeps TinyStories out of the loop and still uses only synthetic tokens; it only limits how much of the generated stream you train on per epoch.

**Expected final checkpoint path:**

`checkpoints/demo_synthetic_v1/demo_synthetic_trained_v1_final.pt`

---

## Generation (Phase 3 controller)

After training completes:

```bash
python run_attractor_llm.py --mode generate \
  --checkpoint checkpoints/demo_synthetic_v1/demo_synthetic_trained_v1_final.pt \
  --prompt "the state is attractor flow cycle" \
  --max-tokens 80 \
  --phase3
```

---

## Example generation output

*(Paste the terminal output from the command above after you train locally.)*

```
<your output here>
```

---

## What this is meant to show

Training with `--dataset synthetic` only proves the **attractor substrate and training loop** can run end-to-end on a compact, procedurally generated token stream with no real-world text. Combined with the generation command above (with `--phase3`), you get a reproducible “synthetic-only” demo path on the `dev_route_1_attractor_v1` line of work.

---

## After training (git)

When you are ready to share weights:

```bash
git add checkpoints/demo_synthetic_v1/ DEMO.md
git commit -m "Add synthetic-only demo checkpoint and DEMO instructions"
git push -u origin demo-synthetic-trained-v1
```

*(Large checkpoints: confirm Git LFS or release artifacts if the repo should not store big binaries.)*
