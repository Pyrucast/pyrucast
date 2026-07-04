"""Python tests for the mesh-transform / 3-D sweep operators:
translate, rotate, sweep_solid, and TRI3 → PENTA6 extrusion."""

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

    out = pyrucast.translate(m, [10.0, 5.0])
    assert out.element_types() == ["TRI3"]
    n0 = out.node(0, 0, 0)
    assert n0.coord() == [10.0, 5.0]
    # Fresh node, distinct from the source; the source is unchanged.
    assert n0.id != ids[0].id
    assert ids[0].coord() == [0.0, 0.0]


def test_rotate_2d_quarter_turn():
    c = pyrucast.Coords(2)
    m = pyrucast.Mesh(c, "TRI3")
    ids = [c.add_node(p) for p in ([1.0, 0.0], [2.0, 0.0], [1.0, 1.0])]
    m.unit().add_cell(ids)

    out = pyrucast.rotate(m, math.pi / 2.0, [0.0, 0.0])
    x, y = out.node(0, 0, 0).coord()
    assert abs(x - 0.0) < 1e-12 and abs(y - 1.0) < 1e-12


def test_rotate_3d_about_z():
    c = pyrucast.Coords(3)
    m, _ = _tri3(c, [[1.0, 0.0, 5.0], [0.0, 1.0, 5.0], [0.0, 0.0, 5.0]])
    out = pyrucast.rotate(m, math.pi / 2.0, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    x, y, z = out.node(0, 0, 0).coord()
    assert abs(x) < 1e-12 and abs(y - 1.0) < 1e-12 and abs(z - 5.0) < 1e-12


def test_extrude_tri3_to_penta6():
    c = pyrucast.Coords(3)
    m, _ = _tri3(c, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    penta = pyrucast.extrude(m, [0.0, 0.0, 2.0], 2)
    assert penta.element_types() == ["PENTA6"]
    assert penta.cell_counts() == [2]


def test_sweep_solid_tri3_to_penta6():
    c = pyrucast.Coords(3)
    a, _ = _tri3(c, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    b, bids = _tri3(c, [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]])

    solid = pyrucast.sweep_solid(a, b, 2)
    assert solid.element_types() == ["PENTA6"]
    assert solid.cell_counts() == [2]
    # The top face of the last layer reuses mesh_b's nodes.
    assert solid.node(0, 1, 3).id == bids[0].id


def test_rotate_via_sweep_builds_a_solid_of_revolution_slice():
    """rotate + sweep_solid together: sweep a face onto its rotated copy."""
    c = pyrucast.Coords(3)
    face, _ = _tri3(c, [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
    rotated = pyrucast.rotate(face, math.pi / 6.0, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    solid = pyrucast.sweep_solid(face, rotated, 1)
    assert solid.element_types() == ["PENTA6"]
    assert solid.cell_counts() == [1]
