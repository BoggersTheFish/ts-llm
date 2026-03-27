"""Light import / smoke regression across phases."""

from __future__ import annotations

import importlib


def test_import_ts_attractor() -> None:
    importlib.import_module("ts_attractor")


def test_import_attractor_llm() -> None:
    importlib.import_module("attractor_llm")
