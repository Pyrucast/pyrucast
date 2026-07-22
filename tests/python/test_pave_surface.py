"""Python tests for the frontal surface mesher `pyrucast.mesher.pave_surface`."""

import math

import pyrucast


def _contour(coords, pts):
    """Build a closed SEG2 contour mesh through `pts` (list of (x, y))."""
    nodes = [coords.add_node([x, y]) for (x, y) in pts]
    mesh = pyrucast.Mesh(coords, "SEG2")
    sm = mesh[0]
    n = len(nodes)
    for i in range(n):
        sm.add_cell([nodes[i], nodes[(i + 1) % n]])
    return mesh


def _total_area(tri):
    """Sum of CCW triangle areas of a single-submesh TRI3 mesh."""
    total = 0.0
    for ci in range(tri.cell_count()):
        p0 = tri.node(0, ci, 0).coord()
        p1 = tri.node(0, ci, 1).coord()
        p2 = tri.node(0, ci, 2).coord()
        a = 0.5 * (
            (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        )
        assert a > 0.0, f"triangle {ci} is not CCW (area {a})"
        total += a
    return total


def test_pave_surface_square_two_triangles():
    c = pyrucast.Coords(2)
    contour = _contour(c, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    tri = pyrucast.mesher.pave_surface(contour, "TRI3", 10.0)
    assert tri.element_types() == ["TRI3"]
    assert tri.cell_count() == 2
    assert abs(_total_area(tri) - 1.0) < 1e-12


def test_pave_surface_circle_fills_interior_and_conserves_area():
    c = pyrucast.Coords(2)
    nseg, r = 32, 4.0
    pts = [
        (r * math.cos(2 * math.pi * i / nseg), r * math.sin(2 * math.pi * i / nseg))
        for i in range(nseg)
    ]
    contour = _contour(c, pts)
    tri = pyrucast.mesher.pave_surface(contour, "TRI3", 0.8)
    assert tri.cell_count() > nseg  # interior nodes were created
    poly_area = 0.5 * nseg * r * r * math.sin(2 * math.pi / nseg)
    assert abs(_total_area(tri) - poly_area) < 1e-6


def test_pave_surface_default_size():
    c = pyrucast.Coords(2)
    contour = _contour(c, [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
    tri = pyrucast.mesher.pave_surface(contour, "TRI3")
    assert abs(_total_area(tri) - 4.0) < 1e-12


def test_pave_surface_qua4_square_is_one_quad():
    c = pyrucast.Coords(2)
    contour = _contour(c, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    q = pyrucast.mesher.pave_surface(contour, "QUA4", 10.0)
    assert q.element_types() == ["QUA4"]
    assert q.cell_count() == 1


def test_pave_surface_qua4_circle_quad_dominant():
    c = pyrucast.Coords(2)
    nseg, r = 32, 4.0
    pts = [
        (r * math.cos(2 * math.pi * i / nseg), r * math.sin(2 * math.pi * i / nseg))
        for i in range(nseg)
    ]
    contour = _contour(c, pts)
    mesh = pyrucast.mesher.pave_surface(contour, "QUA4", 0.8)
    types = mesh.element_types()
    assert "QUA4" in types


def test_pave_surface_3d_square_in_plane():
    c = pyrucast.Coords(3)
    pts = [(0.0, 0.0, 2.0), (4.0, 0.0, 2.0), (4.0, 4.0, 2.0), (0.0, 4.0, 2.0)]
    nodes = [c.add_node(list(p)) for p in pts]
    mesh = pyrucast.Mesh(c, "SEG2")
    sm = mesh[0]
    n = len(nodes)
    for i in range(n):
        sm.add_cell([nodes[i], nodes[(i + 1) % n]])
    tri = pyrucast.mesher.pave_surface(mesh, "TRI3", 1.0)
    # Every output node stays on the plane z = 2.
    for ci in range(tri.cell_count()):
        for ni in range(3):
            assert abs(tri.node(0, ci, ni).coord()[2] - 2.0) < 1e-9


def test_pave_surface_rejects_unsupported_element():
    c = pyrucast.Coords(2)
    contour = _contour(c, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    try:
        pyrucast.mesher.pave_surface(contour, "TET4")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for unsupported element type")
