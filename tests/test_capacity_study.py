"""
Tests for scripts/capacity_study.py (Phase 4).
All tests are fast — no training runs.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _load_study():
    spec = importlib.util.spec_from_file_location(
        "capacity_study", _ROOT / "scripts" / "capacity_study.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Config table structure
# ---------------------------------------------------------------------------

def test_study_configs_has_four_entries() -> None:
    mod = _load_study()
    assert len(mod.STUDY_CONFIGS) == 4


def test_study_configs_ordered_by_fast_dim() -> None:
    mod = _load_study()
    dims = [fd for _, fd, _ in mod.STUDY_CONFIGS]
    assert dims == sorted(dims), "STUDY_CONFIGS must be ordered by increasing fast_dim"


def test_study_configs_names() -> None:
    mod = _load_study()
    names = [n for n, *_ in mod.STUDY_CONFIGS]
    assert names == ["tiny", "small", "base", "large"]


def test_study_configs_slow_dim_grows_with_fast_dim() -> None:
    mod = _load_study()
    for i in range(len(mod.STUDY_CONFIGS) - 1):
        _, fd_a, sd_a = mod.STUDY_CONFIGS[i]
        _, fd_b, sd_b = mod.STUDY_CONFIGS[i + 1]
        assert sd_b > sd_a, "slow_dim must grow with fast_dim"


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def test_count_params_increases_with_capacity() -> None:
    mod = _load_study()
    vocab = 8192
    counts = [mod.count_params(fd, sd, vocab) for _, fd, sd in mod.STUDY_CONFIGS]
    for i in range(len(counts) - 1):
        assert counts[i + 1] > counts[i], (
            f"Expected params to grow: {counts[i]} → {counts[i+1]}"
        )


def test_count_params_tiny_at_vocab_4096() -> None:
    """Pin the actual count at the real tokenizer vocab size (4096)."""
    mod = _load_study()
    n = mod.count_params(64, 16, 4096)
    assert n == 618_144, f"tiny param count changed: expected 618,144 got {n:,}"


def test_count_params_base_at_vocab_4096() -> None:
    mod = _load_study()
    n = mod.count_params(256, 64, 4096)
    assert n == 2_749_056, f"base param count changed: expected 2,749,056 got {n:,}"


# ---------------------------------------------------------------------------
# Comparison table (no crash on missing data)
# ---------------------------------------------------------------------------

def test_print_comparison_table_empty(capsys) -> None:
    mod = _load_study()
    mod.print_comparison_table([])  # must not raise
    out = capsys.readouterr().out
    assert "CAPACITY STUDY RESULTS" in out


def test_print_comparison_table_partial(capsys) -> None:
    mod = _load_study()
    results = [
        {
            "config": "tiny", "fast_dim": 64, "slow_dim": 16,
            "n_params": 1_212_064, "val_ppl": 250.0,
            "h_fast": {"intrinsic_dimensionality": 12.3, "basin_separation_ratio": 1.5},
            "h_slow": {"intrinsic_dimensionality": 4.1, "basin_separation_ratio": None},
        }
    ]
    mod.print_comparison_table(results)
    out = capsys.readouterr().out
    assert "tiny" in out
    assert "250.00" in out


# ---------------------------------------------------------------------------
# Checkpoint path helper
# ---------------------------------------------------------------------------

def test_checkpoint_path_names() -> None:
    mod = _load_study()
    for name, *_ in mod.STUDY_CONFIGS:
        p = mod.checkpoint_path(name)
        assert p.name == f"attractor_{name}.pt"
        assert p.parent.name == "checkpoints"


# ---------------------------------------------------------------------------
# Dry-run smoke test
# ---------------------------------------------------------------------------

def test_dry_run_exits_cleanly(capsys) -> None:
    """--dry-run should print the table and return without training."""
    import argparse
    mod = _load_study()
    # Simulate dry-run by calling print_config_table directly
    mod.print_config_table(vocab_size=8192)
    out = capsys.readouterr().out
    assert "tiny" in out
    assert "small" in out
    assert "base" in out
    assert "large" in out
    assert "params" in out
