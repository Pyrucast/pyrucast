"""Python tests for `pyrucast.mesh.circle`."""

import pyrucast


def test_circle_full_default_seg2():
    c = pyrucast.Coords(2)
    center = c.add_node([0.0, 0.0])
    mesh = pyrucast.mesh.circle(center, [0.0, 0.0, 1.0], 1.0, 8)
    assert mesh.element_types() == ["SEG2"]
    assert mesh.cell_count() == 8


def test_circle_full_seg3():
    c = pyrucast.Coords(2)
    center = c.add_node([0.0, 0.0])
    mesh = pyrucast.mesh.circle(center, [0.0, 0.0, 1.0], 1.0, 6, element_type="SEG3")
    assert mesh.element_types() == ["SEG3"]
    assert mesh.cell_count() == 6


def test_circle_unknown_element_type_raises():
    c = pyrucast.Coords(2)
    center = c.add_node([0.0, 0.0])
    try:
        pyrucast.mesh.circle(center, [0.0, 0.0, 1.0], 1.0, 8, element_type="BOGUS")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown element type")
