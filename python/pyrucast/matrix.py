"""Operators producing a matrix — mirror of ``ops::matrix`` (Rust).

The assemblers proper: stiffness, mass/capacity, geometric stiffness,
consistent tangent, diagonal lumping. Each returns a matrix assembled on the
configuration active at the call; a sum of matrices adds those values.

Assembling a matrix from its blocks alone is not here: it mutates a single
container while preserving its invariant, hence a method
— ``matrix.assemble()``, next to ``matrix.finalize()``.
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
