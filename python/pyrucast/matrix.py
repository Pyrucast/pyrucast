"""Operators producing a matrix — mirror of ``ops::matrix`` (Rust).

The assemblers proper: stiffness, mass/capacity, geometric
géométrique, tangente cohérente, concentration diagonale.

(Re)assembling a matrix from its blocks alone is not here: it mutates a single
container while preserving its invariant, hence a method
— ``matrix.assemble()``, voisine de ``matrix.finalize()``.
"""

from ._pyrucast import (
    geometric,
    lump,
    mass,
    stiffness,
    tangent,
)

__all__ = [
    "geometric",
    "lump",
    "mass",
    "stiffness",
    "tangent",
]
