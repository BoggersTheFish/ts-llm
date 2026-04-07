#!/usr/bin/env python3
"""Merge deterministic template sentences into data/training.json (keeps branch head + metadata).

Warning: a ~300-line corpus makes each training epoch run hundreds of full-sequence forward
passes (see make_training_ids in eval_harness.py). With TRAIN_EPOCHS=1500, training can take
many minutes or hours on CPU. Keep the default five-line corpus for fast demos/tests; run this
script only when you explicitly want a larger training set.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAINING_PATH = ROOT / "data" / "training.json"

ANIMALS = ("cat", "dog", "bird", "fish", "mouse")
PLACES = ("barn", "lake", "sky")
# Closed-class nouns already in the tiny vocab (original five lines + animals + cheese).
NOUNS = ANIMALS + ("cheese",)


def supplement_sentences(rng: random.Random) -> list[str]:
    s: set[str] = set()
    for a in NOUNS:
        for b in NOUNS:
            s.add(f"the {a} chases the {b}")
            s.add(f"the {a} eats the {b}")
    for a in ANIMALS:
        for b in NOUNS:
            s.add(f"the {a} flies to the {b}")
    for a in ANIMALS:
        for b in NOUNS:
            s.add(f"the {a} runs to the {b}")
            s.add(f"the {a} swims to the {b}")
        s.add(f"the {a} flies in the sky")
        s.add(f"the {a} swims in the lake")
    for a in ANIMALS:
        for p in PLACES:
            s.add(f"the {a} runs in the {p}")
            s.add(f"the {a} flies in the {p}")
            s.add(f"the {a} swims in the {p}")
    for a in ANIMALS:
        for p in PLACES:
            s.add(f"the {a} eats in the {p}")
            s.add(f"the {a} chases in the {p}")
    for a in ANIMALS:
        for b in NOUNS:
            s.add(f"the {a} swims in the {b}")
            s.add(f"the {a} flies in the {b}")
    for a in NOUNS:
        for b in NOUNS:
            s.add(f"the {a} flies to the {b}")
    out = sorted(s)
    rng.shuffle(out)
    return out


def main() -> None:
    raw = json.loads(TRAINING_PATH.read_text(encoding="utf-8"))
    head = [str(x) for x in raw["corpus"][: int(raw["branch_line_count"])]]
    canonical5 = [
        "the cat chases the dog",
        "the dog runs to the barn",
        "the bird flies in the sky",
        "the fish swims in the lake",
        "the mouse eats the cheese",
    ]
    if head != canonical5:
        print("Warning: corpus head differs from expected canonical five lines.", file=sys.stderr)

    rng = random.Random(0)
    extra = supplement_sentences(rng)
    merged_corpus = canonical5 + [x for x in extra if x not in set(canonical5)]

    raw["corpus"] = merged_corpus
    if "contrastive_pairs" not in raw:
        raw["contrastive_pairs"] = [
            ["the cat chases the dog", "the cat chases the mouse"],
            ["the cat chases the dog", "the cat chases the bird"],
            ["the mouse eats the cheese", "the mouse eats the dog"],
        ]
    if "contrastive" not in raw:
        raw["contrastive"] = {"lambda": 0.1, "margin": 0.35}

    TRAINING_PATH.write_text(
        json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(merged_corpus)} corpus lines to {TRAINING_PATH}")


if __name__ == "__main__":
    main()
