"""Finite-element spaces — mirror of ``ops::finite_element_space`` (Rust).

The operators producing a ``FiniteElementSpace`` from another: its
restriction to the cells of a mesh.
"""

from ._pyrucast import (
    restrict_fespace as restrict,
)

__all__ = [
    "restrict",
]
