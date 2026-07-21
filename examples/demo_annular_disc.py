"""Annular-disc demo.

Builds two concentric SEG2 circles, fills the disc between them with
TRI3s, assembles all three submeshes into a single Mesh with one colour
per submesh, then opens the interactive viewer.

Run with the `viz-interactive` Python module installed:

    PYO3_PYTHON=/usr/bin/python3.13 \
        maturin develop --features extension-module,viz-interactive
    python examples/demo_annular_disc.py

Interactive controls: left-drag rotates, mouse wheel zooms, key `A`
toggles the X/Y/Z gizmo.
"""

import pyrucast as pc


def main() -> None:
    coords = pc.Coords(2)
    center = coords.add_node([0.0, 0.0])

    # `circle` always takes a 3-D normal, even for a 2-D config:
    # (0, 0, 1) means "circle in the XY plane".
    normal = [0.0, 0.0, 1.0]
    inner_mesh = pc.mesher.circle(center, normal, 1.0, 12)
    outer_mesh = pc.mesher.circle(center, normal, 2.0, 12)

    # Contour for fill_surface: a Mesh holding both SEG2 loops. The CDT
    # detects the outer loop automatically (largest signed area) and
    # treats the inner one as a hole.
    contour = inner_mesh | outer_mesh
    contour = outer_mesh
    surface_mesh = pc.mesher.sweep(inner_mesh, outer_mesh, 2)

    print(surface_mesh)

    # `mesh[i]` returns the i-th submesh handle and shares storage with
    # the parent mesh, so colour changes survive every later
    # add_submesh / __add__.
    inner_mesh[0].face_color = (220, 60, 60)  # red
    outer_mesh[0].face_color = (60, 60, 220)  # blue
    surface_mesh[0].face_color = (60, 180, 60)  # green

    final = pc.Mesh(coords)
    final.add_submesh(inner_mesh[0])
    final.add_submesh(outer_mesh[0])
    final.add_submesh(surface_mesh[0])

    print(final)
    print("submeshes:", final.element_types())
    print("cells per submesh:", final.cell_counts())
    print("len(final):", len(final), "— iterable:", [sm.element_type for sm in final])
    # `mesh[i]` → SubMesh, `submesh[j]` → Cell, `cell[k]` → Node.
    print("first cell of the inner ring:", [n.id for n in final[0][0]])

    final.plot()


if __name__ == "__main__":
    main()
