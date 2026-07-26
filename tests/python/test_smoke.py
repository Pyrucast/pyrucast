"""Smoke test for the pyrucast Python module (Phase 0)."""

import re

import pyrucast


def test_module_importable():
    assert pyrucast is not None


def test_version_exposed():
    assert re.fullmatch(r"\d+\.\d+\.\d+", pyrucast.__version__)
