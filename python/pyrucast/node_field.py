"""Operators producing a field at the nodes — mirror of ``ops::node_field``.

Derivations (a mesh's coordinates, the divergence of a field by elements,
the nodal sum of a per-cell field, restriction, merging) and nodal assembly (imposed flux, and both sides of the
balance `Σ f_int = Σ f_ext` whose gap is the residual). Solving
also produces a nodal field, but keeps its own module:
``pyrucast.solver``.
"""

from ._pyrucast import (
    consolidate_node as consolidate,
)
from ._pyrucast import (
    divergence,
    external_forces,
    internal_forces,
    merge,
    positions,
    restrict,
    restrict_like,
    sum_to_nodes,
)
from ._pyrucast import (
    mask_node as mask,
)

__all__ = [
    "consolidate",
    "divergence",
    "external_forces",
    "internal_forces",
    "mask",
    "merge",
    "positions",
    "restrict",
    "restrict_like",
    "sum_to_nodes",
]
