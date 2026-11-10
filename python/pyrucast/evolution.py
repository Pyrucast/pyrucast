"""Operators producing an ``Evolution`` from other containers — mirror of
``ops::evolution``.

An evolution is mostly declared from its samples (``pyrucast.SubEvolution``,
``pyrucast.Evolution(steps)``); the operators here compute them: ``along``
reads a field along a line of points, one curve per component against the
distance travelled.
"""

from ._pyrucast import along

__all__ = ["along"]
