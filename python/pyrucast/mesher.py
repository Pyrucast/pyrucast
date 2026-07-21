"""Opérateurs de maillage — miroir de ``ops::mesher`` (Rust).

Fabriques et transformations qui produisent ou remanient un maillage :
primitives 1D, balayages, extrusions, transformations rigides, remplissages
surfaciques/volumiques, lecture gmsh, mesures géométriques dérivées.
"""

from ._pyrucast import (
    barycenter as barycenter,
    circle_seg2 as circle_seg2,
    contour as contour,
    elements_on as elements_on,
    extrude as extrude,
    fill_surface as fill_surface,
    from_live_nodes as from_live_nodes,
    line as line,
    merge_nodes as merge_nodes,
    poi1_from_nodes as poi1_from_nodes,
    read_gmsh as read_gmsh,
    read_gmsh_str as read_gmsh_str,
    rotate as rotate,
    surface as surface,
    sweep as sweep,
    sweep_solid as sweep_solid,
    to_poi1 as to_poi1,
    to_quadratic as to_quadratic,
    translate as translate,
    volume as volume,
)

__all__ = [
    "barycenter",
    "circle_seg2",
    "contour",
    "elements_on",
    "extrude",
    "fill_surface",
    "from_live_nodes",
    "line",
    "merge_nodes",
    "poi1_from_nodes",
    "read_gmsh",
    "read_gmsh_str",
    "rotate",
    "surface",
    "sweep",
    "sweep_solid",
    "to_poi1",
    "to_quadratic",
    "translate",
    "volume",
]
