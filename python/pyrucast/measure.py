"""Mesures — miroir de ``ops::measure`` (Rust).

The operators returning a number rather than a container: integral over a
espace éléments finis, normes et produits scalaires globaux.
"""

from ._pyrucast import (
    integral as integral,
    xtx as xtx,
    xty as xty,
)

__all__ = [
    "integral",
    "xtx",
    "xty",
]
