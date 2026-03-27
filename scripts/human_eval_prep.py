#!/usr/bin/env python3
"""Write 20 placeholder blind story slots for human review."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    stories = [{"id": i, "text": f"[placeholder story {i}]"} for i in range(20)]
    Path("human_blind_stories.json").write_text(json.dumps(stories, indent=2), encoding="utf-8")
    print("wrote human_blind_stories.json")


if __name__ == "__main__":
    main()
