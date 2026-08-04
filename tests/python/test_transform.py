"""Python tests for the mesh-transform / 3-D sweep operators: translate,
rotate, the symmetries, sweep_solid, and TRI3 → PENTA6 extrusion."""

import math

import pyrucast


def _tri3(coords, pts):
    """Single-TRI3 mesh from three coordinate triples."""
    ids = [coords.add_node(list(p)) for p in pts]
    m = pyrucast.Mesh(coords, "TRI3")
    m.unit().add_cell(ids)
    return m, ids


def test_translate_fresh_nodes_original_untouched():
    c = pyrucast.Coords(2)
    m = pyrucast.Mesh(c, "TRI3")
    ids = [c.add_node(p) for p in ([0.0, 0.0], [1.0, 0.0], [0.0, 1.0])]
    m.unit().add_cell(ids)

    out = pyrucast.mesh.translate(m, [10.0, 5.0])
    assert out.element_types() == ["TRI3"]
    n0 = out.node(0, 0, 0)
    assert n0.position() == [10.0, 5.0]
    # Fresh node, distinct from the source; the source is unchanged.
    assert n0.id != ids[0].id
    assert ids[0].position() == [0.0, 0.0]


def test_rotate_2d_quarter_turn():
    c = pyrucast.Coords(2)
    m = pyrucast.Mesh(c, "TRI3")
    ids = [c.add_node(p) for p in ([1.0, 0.0], [2.0, 0.0], [1.0, 1.0])]
    m.unit().add_cell(ids)

    out = pyrucast.mesh.rotate(m, math.pi / 2.0, [0.0, 0.0])
    x, y = out.node(0, 0, 0).position()
    assert abs(x - 0.0) < 1e-12 and abs(y - 1.0) < 1e-12


def test_rotate_3d_about_z():
    c = pyrucast.Coords(3)
    m, _ = _tri3(c, [[1.0, 0.0, 5.0], [0.0, 1.0, 5.0], [0.0, 0.0, 5.0]])
    out = pyrucast.mesh.rotate(m, math.pi / 2.0, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    x, y, z = out.node(0, 0, 0).position()
    assert abs(x) < 1e-12 and abs(y - 1.0) < 1e-12 and abs(z - 5.0) < 1e-12


def test_extrude_tri3_to_penta6():
    c = pyrucast.Coords(3)
    m, _ = _tri3(c, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    penta = pyrucast.mesh.extrude(m, [0.0, 0.0, 2.0], 2)
    assert penta.element_types() == ["PENTA6"]
    assert penta.cell_counts() == [2]


def test_sweep_solid_tri3_to_penta6():
    c = pyrucast.Coords(3)
    a, _ = _tri3(c, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    b, bids = _tri3(c, [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]])

    solid = pyrucast.mesh.sweep_solid(a, b, 2)
    assert solid.element_types() == ["PENTA6"]
    assert solid.cell_counts() == [2]
    # The top face of the last layer reuses mesh_b's nodes.
    assert solid.node(0, 1, 3).id == bids[0].id


def test_rotate_via_sweep_builds_a_solid_of_revolution_slice():
    """rotate + sweep_solid together: sweep a face onto its rotated copy."""
    c = pyrucast.Coords(3)
    face, _ = _tri3(c, [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
    rotated = pyrucast.mesh.rotate(
        face, math.pi / 6.0, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]
    )
    solid = pyrucast.mesh.sweep_solid(face, rotated, 1)
    assert solid.element_types() == ["PENTA6"]
    assert solid.cell_counts() == [1]


def _signed_volume(mesh):
    """Signed volume of a TET4 cell, positive for the direct node ordering."""
    p = [mesh.node(0, 0, i).position() for i in range(4)]
    e = [[p[k][j] - p[0][j] for j in range(3)] for k in (1, 2, 3)]
    return (
        e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1])
        - e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0])
        + e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0])
    ) / 6.0


def _tet4(coords, pts):
    """Single-TET4 mesh from four coordinate triples."""
    ids = [coords.add_node(list(p)) for p in pts]
    m = pyrucast.Mesh(coords, "TET4")
    m.unit().add_cell(ids)
    return m, ids


def test_symmetry_point_2d_is_a_half_turn():
    c = pyrucast.Coords(2)
    m = pyrucast.Mesh(c, "TRI3")
    ids = [c.add_node(p) for p in ([1.0, 0.0], [2.0, 0.0], [1.0, 1.0])]
    m.unit().add_cell(ids)

    out = pyrucast.mesh.symmetry_point(m, [0.0, 0.0])
    assert out.node(0, 0, 0).position() == [-1.0, 0.0]
    assert out.node(0, 0, 2).position() == [-1.0, -1.0]
    # Fresh nodes, source untouched.
    assert out.node(0, 0, 0).id != ids[0].id
    assert ids[0].position() == [1.0, 0.0]


def test_symmetry_plane_3d_keeps_the_jacobian_positive():
    c = pyrucast.Coords(3)
    m, _ = _tet4(
        c,
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 2.0]],
    )
    assert _signed_volume(m) > 0.0

    # Mirror through z = 0 (normal deliberately unnormalized).
    out = pyrucast.mesh.symmetry_plane(m, [0.0, 0.0, 0.0], [0.0, 0.0, 3.0])
    assert _signed_volume(out) > 0.0
    assert sorted(out.node(0, 0, i).position() for i in range(4)) == [
        [0.0, 0.0, -2.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, -1.0],
        [1.0, 0.0, -1.0],
    ]


def test_symmetry_line_3d_matches_a_half_turn():
    c = pyrucast.Coords(3)
    m, _ = _tri3(c, [[1.0, 0.0, 2.0], [2.0, 0.0, 2.0], [1.0, 1.0, 2.0]])

    out = pyrucast.mesh.symmetry_line(m, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    turned = pyrucast.mesh.rotate(m, math.pi, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    for i in range(3):
        a = out.node(0, 0, i).position()
        b = turned.node(0, 0, i).position()
        assert all(abs(x - y) < 1e-12 for x, y in zip(a, b))


def test_symmetry_is_also_a_mesh_method():
    c = pyrucast.Coords(2)
    m, _ = _tri3(c, [[0.0, 1.0], [1.0, 1.0], [0.0, 3.0]])
    # The line y = x: by two points, by its normal, and via the methods.
    by_line = m.symmetry_line([0.0, 0.0], [1.0, 1.0])
    by_plane = pyrucast.mesh.symmetry_plane(m, [0.0, 0.0], [1.0, -1.0])
    for i in range(3):
        a = by_line.node(0, 0, i).position()
        b = by_plane.node(0, 0, i).position()
        assert all(abs(x - y) < 1e-12 for x, y in zip(a, b))
    assert m.symmetry_point([0.0, 0.0]).node(0, 0, 0).position() == [0.0, -1.0]
