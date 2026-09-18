"""`border`: a surface's border, with optional splitting into edges by angle."""

import pyrucast


def _unit_square_quad():
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    d = c.add_node([1.0, 1.0])
    e = c.add_node([0.0, 1.0])
    mesh = pyrucast.Mesh(c, "QUA4")
    mesh[0].add_cell([a, b, d, e])
    return mesh


def test_border_default_is_one_closed_loop():
    mesh = _unit_square_quad()
    b = pyrucast.mesh.border(mesh)
    assert len(b) == 1
    assert b.element_types() == ["SEG2"]
    assert b.cell_counts() == [4]  # boucle fermée : 4 segments


def test_border_angle_splits_into_open_aretes():
    mesh = _unit_square_quad()
    # Four corners at 90° → four open edges of one segment each.
    b = pyrucast.mesh.border(mesh, angle_deg=45.0)
    assert len(b) == 4
    assert b.cell_counts() == [1, 1, 1, 1]
