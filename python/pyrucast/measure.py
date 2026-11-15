"""Measures — mirror of ``ops::measure`` (Rust).

The operators returning a number rather than a container: integral over a
finite element space, global norms and scalar products.
"""

from ._pyrucast import (
    integral,
    xtx,
    xty,
)

__all__ = [
    "integral",
    "xtx",
    "xty",
]
