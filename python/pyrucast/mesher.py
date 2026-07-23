"""Opérateurs de maillage — miroir de ``ops::mesher`` (Rust).

Fabriques et transformations qui produisent ou remanient un maillage :
primitives 1D, balayages, extrusions, transformations rigides, remplissages
surfaciques/volumiques, lecture gmsh, mesures géométriques dérivées.
"""

from ._pyrucast import (
    arc as arc,
    barycenter as barycenter,
    circle as circle,
    contour as contour,
    elements_on as elements_on,
    extrude as extrude,
    from_live_nodes as from_live_nodes,
    invert as invert,
    line as line,
    merge_nodes as merge_nodes,
    orient as orient,
    poi1_from_nodes as poi1_from_nodes,
    read_gmsh as read_gmsh,
    read_gmsh_str as read_gmsh_str,
    rotate as rotate,
    sweep as sweep,
    sweep_solid as sweep_solid,
    to_poi1 as to_poi1,
    to_quadratic as to_quadratic,
    transfinite as transfinite,
    translate as translate,
    triangulate_surface as triangulate_surface,
    volume as volume,
)

__all__ = [
    "arc",
    "barycenter",
    "circle",
    "contour",
    "elements_on",
    "extrude",
    "from_live_nodes",
    "invert",
    "line",
    "merge_nodes",
    "orient",
    "poi1_from_nodes",
    "read_gmsh",
    "read_gmsh_str",
    "rotate",
    "sweep",
    "sweep_solid",
    "to_poi1",
    "to_quadratic",
    "transfinite",
    "translate",
    "triangulate_surface",
    "volume",
]
