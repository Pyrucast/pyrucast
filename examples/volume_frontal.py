"""Hexahedral boundary layer over a tetrahedral core (`pyrucast.mesh.pave_volume`).

Builds a box, takes its skin, and meshes the inside with hexahedra against the
boundary and tetrahedra in the middle — the two joined by pyramids, the one
element that presents a square on one side and triangles on the other.

Contrast with `triangulate_volume`, which fills the whole solid with
tetrahedra: the boundary layer is where stress and flux gradients are steepest
and where an element's shape decides the accuracy.

Run:

    PYO3_PYTHON=/usr/bin/python3.13 \
        maturin develop --features extension-module
    python examples/volume_frontal.py
"""

import pyrucast as pc


def box_skin(n: int) -> pc.Mesh:
    """The skin of an n³ box of hexahedra: closed QUA4, normals outward."""
    coords = pc.Coords(3)
    a = coords.add_node([0.0, 0.0, 0.0])
    b = coords.add_node([1.0, 0.0, 0.0])
    c = coords.add_node([1.0, 0.0, 1.0])
    d = coords.add_node([0.0, 0.0, 1.0])
    # Wound so the face's normal is +y, the direction it gets extruded in;
    # the other way round every hexahedron — and so the skin — comes out
    # inside out.
    ring = None
    for p, q in ((a, d), (d, c), (c, b), (b, a)):
        seg = pc.mesh.line(p, q, n)
        ring = seg if ring is None else ring | seg
    face = pc.mesh.pave_surface(pc.mesh.consolidate(ring), "QUA4", all_quad=True)
    return pc.mesh.skin(pc.mesh.extrude(face, [0.0, 1.0, 0.0], n))


def main() -> None:
    skin = box_skin(6)
    print("peau       :", dict(zip(skin.element_types(), skin.cell_counts())))

    # One layer of hexahedra, then tetrahedra for the rest.
    mesh = pc.mesh.pave_volume(skin, layers=1, thickness=0.08, size=0.2)
    print("pave_volume:", dict(zip(mesh.element_types(), mesh.cell_counts())))

    # Two layers: the front advances again on what it left behind.
    thick = pc.mesh.pave_volume(skin, layers=2, thickness=0.06, size=0.2)
    print("2 couches  :", dict(zip(thick.element_types(), thick.cell_counts())))

    # For comparison, the all-tetrahedron mesher on the same skin. It needs a
    # TRI3 envelope, hence the conversion, and `allow_surface_nodes` so it may
    # cut the envelope's facets finer where it cannot fit them as they are.
    tets = pc.mesh.triangulate_volume(
        pc.mesh.convert(skin, "TRI3"), 0.2, allow_surface_nodes=True
    )
    print("tout tétra :", dict(zip(tets.element_types(), tets.cell_counts())))


if __name__ == "__main__":
    main()
