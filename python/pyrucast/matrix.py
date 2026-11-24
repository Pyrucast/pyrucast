"""Operators producing a matrix — mirror of ``ops::matrix`` (Rust).

The assemblers proper: stiffness, mass/capacity, geometric stiffness,
consistent tangent, diagonal lumping. Each returns a matrix assembled on the
configuration active at the call; a sum of matrices adds those values. And the
projection of a matrix on a list of modes, ``project_on_basis``.

Assembling a matrix from its blocks alone is not here: it mutates a single
container while preserving its invariant, hence a method
— ``matrix.assemble()``.
"""

from ._pyrucast import (
    geometric,
    lump,
    mass,
    stiffness,
    tangent,
)
from ._pyrucast import (
    project_on_basis_matrix as project_on_basis,
)

__all__ = [
    "geometric",
    "lump",
    "mass",
    "project_on_basis",
    "stiffness",
    "tangent",
]
