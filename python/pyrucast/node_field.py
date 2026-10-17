"""Operators producing a field at the nodes — mirror of ``ops::node_field``.

Derivations (a mesh's coordinates, the divergence of a field by elements,
the nodal sum of a per-cell field, restriction, merging) and nodal assembly (imposed flux, and both sides of the
balance `Σ f_int = Σ f_ext` whose gap is the residual). Solving
also produces a nodal field, but keeps its own module:
``pyrucast.solver``.
"""

from ._pyrucast import (
    mask_node as mask,
    consolidate_node as consolidate,
    positions as positions,
    divergence as divergence,
    external_forces as external_forces,
    internal_forces as internal_forces,
    merge as merge,
    restrict as restrict,
    restrict_like as restrict_like,
    sum_to_nodes as sum_to_nodes,
)

__all__ = [
    "mask",
    "consolidate",
    "positions",
    "divergence",
    "external_forces",
    "internal_forces",
    "merge",
    "restrict",
    "restrict_like",
    "sum_to_nodes",
]
