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


def test_spill_stats_reports_an_inactive_allocator():
    """The test suite runs without `PYRUCAST_SPILL_DIR`, so nothing spills.

    A binary that carries `spill` still answers, with a threshold of `None`;
    one that does not carries the same keys, so a script never has to ask.
    """
    stats = pyrucast.spill_stats()
    assert set(stats) == {"threshold", "count", "largest", "live", "peak"}
    assert stats["threshold"] is None
    assert stats["count"] == 0
    assert stats["live"] == 0 and stats["peak"] == 0
