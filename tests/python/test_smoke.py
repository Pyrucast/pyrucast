"""Smoke test for the pyrucast Python module (Phase 0)."""

import re

import pyrucast


def test_module_importable():
    assert pyrucast is not None


def test_version_exposed():
    assert re.fullmatch(r"\d+\.\d+\.\d+", pyrucast.__version__)


def test_features_exposed():
    """`__features__` says what this binary carries, not what it could carry.

    The published wheels compile `viz`; the sdist does not. Without this constant,
    difference only shows in the `AttributeError` that `plot()` raises.
    """
    # ANCHOR: features
    features = pyrucast.__features__
    assert isinstance(features, tuple)
    assert features, "an imported module compiles at least extension-module"
    assert all(isinstance(f, str) for f in features)
    assert "extension-module" in features
    # ANCHOR_END: features


def test_features_match_what_is_compiled():
    """The list is observed on the API, not copied by hand."""
    assert ("viz" in pyrucast.__features__) == hasattr(pyrucast.Mesh, "plot")
