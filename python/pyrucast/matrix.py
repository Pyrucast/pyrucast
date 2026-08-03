"""Opérateurs produisant une matrice — miroir de ``ops::matrix`` (Rust).

Les assembleurs proprement dits : rigidité, masse/capacité, rigidité
géométrique, tangente cohérente, concentration diagonale.
"""

from ._pyrucast import (
    assemble as assemble,
    geometric as geometric,
    lump as lump,
    mass as mass,
    stiffness as stiffness,
    tangent as tangent,
)

__all__ = [
    "assemble",
    "geometric",
    "lump",
    "mass",
    "stiffness",
    "tangent",
]
