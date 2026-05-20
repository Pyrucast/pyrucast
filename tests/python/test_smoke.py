"""Smoke test for the pyrucast Python module (Phase 0)."""

import pyrucast


def test_module_importable():
    assert pyrucast is not None


def test_version_exposed():
    assert pyrucast.__version__ == "0.0.0"
