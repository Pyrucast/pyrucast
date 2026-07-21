"""Python tests for `pyrucast.mesher.circle` (full circle and arc forms)."""

import math

import pyrucast


def test_circle_full_default_seg2():
    c = pyrucast.Coords(2)
    center = c.add_node([0.0, 0.0])
    mesh = pyrucast.mesher.circle(center, [0.0, 0.0, 1.0], 1.0, 8)
    assert mesh.element_types() == ["SEG2"]
    assert mesh.cell_count() == 8


def test_circle_full_seg3():
    c = pyrucast.Coords(2)
    center = c.add_node([0.0, 0.0])
    mesh = pyrucast.mesher.circle(center, [0.0, 0.0, 1.0], 1.0, 6, element_type="SEG3")
    assert mesh.element_types() == ["SEG3"]
    assert mesh.cell_count() == 6


def test_circle_arc_default_seg2():
    c = pyrucast.Coords(2)
    center = c.add_node([0.0, 0.0])
    a = c.add_node([1.0, 0.0])
    b = c.add_node([0.0, 1.0])
    mesh = pyrucast.mesher.circle(a, center, b, 3)
    assert mesh.element_types() == ["SEG2"]
    assert mesh.cell_count() == 3
    assert mesh.node(0, 0, 0).id == a.id
    assert mesh.node(0, 2, 1).id == b.id


def test_circle_arc_seg3():
    c = pyrucast.Coords(2)
    center = c.add_node([0.0, 0.0])
    a = c.add_node([1.0, 0.0])
    b = c.add_node([0.0, 1.0])
    mesh = pyrucast.mesher.circle(a, center, b, 2, element_type="SEG3")
    assert mesh.element_types() == ["SEG3"]
    assert mesh.cell_count() == 2


def test_circle_arc_nodes_lie_on_circle():
    c = pyrucast.Coords(2)
    center = c.add_node([1.0, 1.0])
    a = c.add_node([3.0, 1.0])
    b = c.add_node([1.0, 3.0])
    mesh = pyrucast.mesher.circle(a, center, b, 4)
    for ei in range(mesh.cell_count()):
        for corner in range(2):
            x, y = mesh.node(0, ei, corner).coord()
            dist = math.hypot(x - 1.0, y - 1.0)
            assert abs(dist - 2.0) < 1e-10


def test_circle_arc_rejects_unequal_radii():
    c = pyrucast.Coords(2)
    center = c.add_node([0.0, 0.0])
    a = c.add_node([1.0, 0.0])
    b = c.add_node([0.0, 2.0])
    try:
        pyrucast.mesher.circle(a, center, b, 3)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for unequal radii")


def test_circle_unknown_element_type_raises():
    c = pyrucast.Coords(2)
    center = c.add_node([0.0, 0.0])
    try:
        pyrucast.mesher.circle(center, [0.0, 0.0, 1.0], 1.0, 8, element_type="BOGUS")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown element type")
