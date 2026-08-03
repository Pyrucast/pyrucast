"""Opérateurs produisant une matrice — miroir de ``ops::matrix`` (Rust).

Les assembleurs proprement dits : rigidité, masse/capacité, rigidité
géométrique, tangente cohérente, concentration diagonale.

La (ré)assemblage d'une matrice depuis ses seuls blocs n'est pas ici : elle
mute un unique conteneur en préservant son invariant, c'est donc une méthode
— ``matrix.assemble()``, voisine de ``matrix.finalize()``.
"""

from ._pyrucast import (
    geometric as geometric,
    lump as lump,
    mass as mass,
    stiffness as stiffness,
    tangent as tangent,
)

__all__ = [
    "geometric",
    "lump",
    "mass",
    "stiffness",
    "tangent",
]
