"""Mesures — miroir de ``ops::measure`` (Rust).

Les opérateurs qui rendent un nombre et non un conteneur : intégrale sur un
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
