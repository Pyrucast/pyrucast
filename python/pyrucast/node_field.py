"""Operators producing a field at the nodes — mirror of ``ops::node_field``.

Derivations (a mesh's coordinates, the divergence of a field by elements,
a field at the Gauss points brought to the nodes, restriction, projection
onto another mesh, merging, forces projected on a list of modes and modal
coordinates recombined) and nodal assembly (imposed flux, and both sides of the
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
    project,
    recombine,
    to_nodes,
)
from ._pyrucast import (
    mask_node as mask,
)
from ._pyrucast import (
    project_on_basis_node as project_on_basis,
)
from ._pyrucast import (
    restrict_like_node as restrict_like,
)
from ._pyrucast import (
    restrict_node as restrict,
)

__all__ = [
    "consolidate",
    "divergence",
    "external_forces",
    "internal_forces",
    "mask",
    "merge",
    "positions",
    "project",
    "project_on_basis",
    "recombine",
    "restrict",
    "restrict_like",
    "to_nodes",
]
