Example tiny checkpoint for repository demos.

File:
- `attractor_llm_tiny_example.pt` (trained on `data/train.txt`, CPU, 1 epoch)

Recreate command:

```bash
.venv/bin/python run_attractor_llm.py --mode train --dataset custom --data-file data/train.txt \
  --epochs 1 --state-dim 16 --dynamics full --seq-len 8 --batch-size 1 --lr 0.001 \
  --grad-clip 1.0 --no-tiktoken --eval-every 0 --device cpu --checkpoint-dir examples/model_artifacts \
  --save-every 1 --seed 42 --no-train-progress
```

Quick generation check:

```bash
.venv/bin/python run_attractor_llm.py "Once upon a time" \
  --checkpoint examples/model_artifacts/attractor_llm_tiny_example.pt --max-tokens 20 --device cpu
```
