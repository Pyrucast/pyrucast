"""Operators producing a mesh — mirror of ``ops::mesh`` (Rust).

Meshers (line, circle, arc, transfinite, paving, triangulation), sweeps and
transformations, skin and border extraction, geometric selections, gmsh
reading. Everything that returns a ``Mesh`` is here, whatever the input —
including ``select``, which extracts a field's support.

``from_gmsh`` and ``from_medcoupling`` are the only functions of this module
written in Python: they need an interpreter carrying the ``gmsh`` or
``medcoupling`` module, which Rust cannot have — and they import it only when
called, never at ``import pyrucast``. They merely fetch the arrays of a live
gmsh model or of a MED file and hand them to ``from_arrays``, the Rust import
every exchange format shares.
"""

from ._pyrucast import (
    arc,
    barycenter,
    border,
    chain,
    circle,
    cleanup,
    convert,
    copy,
    element_type_from_gmsh,
    elements_on,
    extrude,
    from_arrays,
    from_live_nodes,
    gauss_to_external,
    grid_surface,
    grid_surface2,
    invert,
    line,
    match_gauss,
    merge_nodes,
    merge_triangles,
    orient,
    pave_surface,
    pave_volume,
    poi1_from_nodes,
    points_below_plane,
    points_in_cone,
    points_in_cylinder,
    points_in_sphere,
    points_in_torus,
    points_on_cone,
    points_on_cylinder,
    points_on_line,
    points_on_plane,
    points_on_sphere,
    points_on_torus,
    read_gmsh,
    read_gmsh_str,
    regularize,
    revolve,
    rotate,
    select,
    skin,
    sweep,
    sweep_solid,
    symmetry_line,
    symmetry_plane,
    symmetry_point,
    to_poi1,
    to_quadratic,
    transfinite,
    translate,
    triangulate_surface,
    triangulate_volume,
)
from ._pyrucast import (
    consolidate_mesh as consolidate,
)

__all__ = [
    "arc",
    "barycenter",
    "border",
    "chain",
    "circle",
    "cleanup",
    "consolidate",
    "convert",
    "copy",
    "element_type_from_gmsh",
    "elements_on",
    "extrude",
    "from_arrays",
    "from_gmsh",
    "from_live_nodes",
    "from_medcoupling",
    "gauss_to_external",
    "grid_surface",
    "grid_surface2",
    "invert",
    "line",
    "match_gauss",
    "merge_nodes",
    "merge_triangles",
    "orient",
    "pave_surface",
    "pave_volume",
    "poi1_from_nodes",
    "points_below_plane",
    "points_in_cone",
    "points_in_cylinder",
    "points_in_sphere",
    "points_in_torus",
    "points_on_cone",
    "points_on_cylinder",
    "points_on_line",
    "points_on_plane",
    "points_on_sphere",
    "points_on_torus",
    "read_gmsh",
    "read_gmsh_str",
    "regularize",
    "revolve",
    "rotate",
    "select",
    "skin",
    "sweep",
    "sweep_solid",
    "symmetry_line",
    "symmetry_plane",
    "symmetry_point",
    "to_poi1",
    "to_quadratic",
    "transfinite",
    "translate",
    "triangulate_surface",
    "triangulate_volume",
]


# Exchange adapters written in Python: they need an interpreter carrying the
# `gmsh` / `medcoupling` modules, imported only when one of them runs.
from ._gmsh import from_gmsh
from ._med import from_medcoupling
