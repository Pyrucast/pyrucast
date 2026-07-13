"""Frontal surface mesher demo (`pyrucast.mesher.surface`).

Builds a circular SEG2 contour and fills its interior with a
size-controlled advancing front that creates interior nodes — first as
triangles (TRI3), then as a quad-dominant mesh (QUA4). Unlike
`fill_surface` (which only triangulates the contour nodes), `surface`
honours a target element size.

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
    contour = pc.mesher.circle_seg2(center, [0.0, 0.0, 1.0], 5.0, 48)

    # Triangles of edge length ~0.8 (interior nodes are created).
    tri = pc.mesher.surface(contour, "TRI3", 0.8)
    print("TRI3 :", tri.element_types(), tri.cell_count(), "cells")

    # Quad-dominant variant (may carry a few triangles).
    quad = pc.mesher.surface(contour, "QUA4", 0.8)
    print("QUA4 :", quad.element_types(), quad.cell_count(), "cells")


if __name__ == "__main__":
    main()
