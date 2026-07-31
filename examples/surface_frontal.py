"""Frontal quadrangle mesher demo (`pyrucast.mesher.pave_surface`).

Builds a circular SEG2 contour and paves its interior with quadrangles laid
in rows walking inward from the boundary. Contrast with
`triangulate_surface`, which triangulates and can only recombine triangles
into quadrangles two by two afterwards.

Run:

    PYO3_PYTHON=/usr/bin/python3.13 \
        maturin develop --features extension-module
    python examples/surface_frontal.py
"""

import pyrucast as pc


def main() -> None:
    coords = pc.Coords(2)
    center = coords.add_node([0.0, 0.0])

    # Circular contour: radius 5, 48 segments, in the XY plane.
    contour = pc.mesher.circle(center, [0.0, 0.0, 1.0], 5.0, 48)

    # Quadrangles of edge length ~0.8; a few triangles may remain.
    quad = pc.mesher.pave_surface(contour, "QUA4", 0.8)
    print("QUA4      :", quad.element_types(), quad.cell_count(), "cells")

    # Asking for it outright: no triangle at all. The contour has an even
    # number of segments, so this costs nothing here.
    strict = pc.mesher.pave_surface(contour, "QUA4", 0.8, all_quad=True)
    print("all_quad  :", strict.element_types(), strict.cell_count(), "cells")

    # Compare with the Delaunay mesher's recombination on the same contour.
    recombined = pc.mesher.triangulate_surface(contour, "QUA4", 0.8)
    print("recombined:", recombined.element_types(), recombined.cell_count(), "cells")


if __name__ == "__main__":
    main()
