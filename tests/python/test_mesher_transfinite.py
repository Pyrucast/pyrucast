"""Python tests for `pyrucast.mesh.transfinite` (the DALL equivalent)."""

import pyrucast


def _rectangle(nx, ny):
    c = pyrucast.Coords(2)
    p0 = c.add_node([0.0, 0.0])
    p1 = c.add_node([2.0, 0.0])
    p2 = c.add_node([2.0, 1.0])
    p3 = c.add_node([0.0, 1.0])
    side1 = pyrucast.mesh.line(p0, p1, nx)
    side2 = pyrucast.mesh.line(p1, p2, ny)
    side3 = pyrucast.mesh.line(p2, p3, nx)
    side4 = pyrucast.mesh.line(p3, p0, ny)
    return side1, side2, side3, side4


def test_transfinite_default_qua4():
    side1, side2, side3, side4 = _rectangle(4, 2)
    mesh = pyrucast.mesh.transfinite(side1, side2, side3, side4)
    assert mesh.element_types() == ["QUA4"]
    assert mesh.cell_count() == 8


def test_transfinite_tri3():
    side1, side2, side3, side4 = _rectangle(4, 2)
    mesh = pyrucast.mesh.transfinite(side1, side2, side3, side4, element_type="TRI3")
    assert mesh.element_types() == ["TRI3"]
    assert mesh.cell_count() == 16


def test_transfinite_boundary_matches_input_corners():
    side1, side2, side3, side4 = _rectangle(3, 2)
    mesh = pyrucast.mesh.transfinite(side1, side2, side3, side4)
    assert mesh.node(0, 0, 0).id == side1.node(0, 0, 0).id
    assert mesh.node(0, 0, 1).id == side1.node(0, 1, 0).id


def test_transfinite_rejects_mismatched_opposite_sides():
    c = pyrucast.Coords(2)
    p0 = c.add_node([0.0, 0.0])
    p1 = c.add_node([2.0, 0.0])
    p2 = c.add_node([2.0, 1.0])
    p3 = c.add_node([0.0, 1.0])
    side1 = pyrucast.mesh.line(p0, p1, 4)
    side2 = pyrucast.mesh.line(p1, p2, 2)
    side3 = pyrucast.mesh.line(p2, p3, 5)  # ≠ side1
    side4 = pyrucast.mesh.line(p3, p0, 2)
    try:
        pyrucast.mesh.transfinite(side1, side2, side3, side4)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for mismatched opposite sides")


def test_transfinite_unknown_element_type_raises():
    side1, side2, side3, side4 = _rectangle(2, 2)
    try:
        pyrucast.mesh.transfinite(side1, side2, side3, side4, element_type="BOGUS")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown element type")
