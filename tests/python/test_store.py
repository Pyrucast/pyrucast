"""Python tests for the global store (Phase 1) — set_swap_dir / swap_dir bindings."""

from pathlib import Path

import pyrucast


def test_set_swap_dir_round_trip(tmp_path):
    pyrucast.set_swap_dir(tmp_path)
    assert Path(pyrucast.swap_dir()) == tmp_path


def test_swap_dir_changeable(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    pyrucast.set_swap_dir(first)
    assert Path(pyrucast.swap_dir()) == first
    pyrucast.set_swap_dir(second)
    assert Path(pyrucast.swap_dir()) == second
