#!/usr/bin/env python3
"""Prepare small tokenized arrays from **local** TinyStories JSON (no network).

Writes ``data/tinystories/processed/{train,val}_tokens.npy`` and ``data/eval_prompts_50.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from attractor_llm.tokenizer import AttractorTokenizer


def main() -> None:
    extracted = REPO_ROOT / "data" / "tinystories" / "extracted"
    out_dir = REPO_ROOT / "data" / "tinystories" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AttractorTokenizer(use_tiktoken=False, vocab_cap=256)
    files = sorted(extracted.glob("*.json"))[:2]
    if not files:
        raise SystemExit("No JSON under data/tinystories/extracted")
    ids: list[int] = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data[:50]:
                if isinstance(item, dict):
                    story = item.get("story", "")
                else:
                    story = str(item)
                if story:
                    ids.extend(tok.encode(story)[:200])
    n = len(ids)
    split = int(n * 0.95)
    train = np.array(ids[:split], dtype=np.int32)
    val = np.array(ids[split:], dtype=np.int32)
    np.save(out_dir / "train_tokens.npy", train)
    np.save(out_dir / "val_tokens.npy", val)
    prompts = [
        {"id": i, "text": f"Hybrid plot A{i} meets plot B{i} in the forest."}
        for i in range(50)
    ]
    (REPO_ROOT / "data" / "eval_prompts_50.json").write_text(
        json.dumps(prompts, indent=2),
        encoding="utf-8",
    )
    print("wrote", out_dir / "train_tokens.npy", len(train), "ids")
    print("wrote data/eval_prompts_50.json")


if __name__ == "__main__":
    main()
