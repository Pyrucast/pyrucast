"""Rigid copies and 3-D sweep demo (`translate`, `rotate`, `sweep_solid`).

Meshes a small surface, then shows the three new mesh-transform operators:

- `translate` / `rotate` return a fresh copy of a mesh with its own new
  nodes (the source is left untouched);
- `sweep_solid` is the 3-D companion of `sweep_qua4`: it links two matching
  surfaces into a solid — TRI3 faces become PENTA6 prisms, QUA4 faces HEX8.

Combining `rotate` with `sweep_solid` builds one angular slice of a solid
of revolution; combining `translate` with `sweep_solid` extrudes a surface
onto a shifted copy (equivalent to a straight `extrude`).

Run:

    PYO3_PYTHON=/usr/bin/python3.13 \
        maturin develop --features extension-module
    python examples/revolution_penta6.py
"""

import math

import pyrucast as pc


def main() -> None:
    coords = pc.Coords(3)

    # A flat TRI3 surface in the plane y = 0, offset from the z axis so a
    # rotation about z sweeps a genuine wedge of material.
    center = coords.add_node([1.5, 0.0, 0.5])
    contour = pc.mesher.circle_seg2(center, [0.0, 1.0, 0.0], 0.5, 24)
    face = pc.mesher.surface(contour, "TRI3", 0.25)
    print("face  :", face.element_types(), face.cell_count(), "cells")

    # One 15° slice of a solid of revolution about the z axis.
    rotated = pc.mesher.rotate(face, math.radians(15.0), [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    wedge = pc.mesher.sweep_solid(face, rotated, 2)
    print("wedge :", wedge.element_types(), wedge.cell_count(), "cells")

    # A straight prism block: sweep the face onto a translated copy.
    shifted = pc.mesher.translate(face, [0.0, 3.0, 0.0])
    block = pc.mesher.sweep_solid(face, shifted, 3)
    print("block :", block.element_types(), block.cell_count(), "cells")

    # The source face is untouched by either copy.
    print("source cells still:", face.cell_count())


if __name__ == "__main__":
    main()
