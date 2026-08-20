"""Smoke test for the pyrucast Python module (Phase 0)."""

import re

import pyrucast


def test_module_importable():
    assert pyrucast is not None


def test_version_exposed():
    assert re.fullmatch(r"\d+\.\d+\.\d+", pyrucast.__version__)


def test_features_exposed():
    """`__features__` dit ce que ce binaire porte, pas ce qu'il pourrait porter.

    Les wheels publiées compilent `viz` ; la sdist non. Sans cette constante, la
    différence ne se lit qu'à l'`AttributeError` que lève `plot()`.
    """
    # ANCHOR: features
    features = pyrucast.__features__
    assert isinstance(features, tuple)
    assert features, "un module importé compile au moins extension-module"
    assert all(isinstance(f, str) for f in features)
    assert "extension-module" in features
    # ANCHOR_END: features


def test_features_match_what_is_compiled():
    """La liste est constatée sur l'API, pas recopiée à la main."""
    assert ("viz" in pyrucast.__features__) == hasattr(pyrucast.Mesh, "plot")
