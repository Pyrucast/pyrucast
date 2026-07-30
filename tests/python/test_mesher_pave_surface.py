"""Python tests for `pyrucast.mesher.pave_surface`."""

import math

import pytest

import pyrucast as pc


def _closed_loop(coords, points):
    """A closed SEG2 loop through `points`, in the order given.

    `pave_surface` wants each loop in a single submesh, so the per-segment
    meshes are consolidated exactly as the formation script does.
    """
    nodes = [coords.add_node(list(p)) for p in points]
    mesh = None
    for a, b in zip(nodes, nodes[1:] + nodes[:1]):
        seg = pc.mesher.line(a, b, 1)
        mesh = seg if mesh is None else mesh | seg
    return pc.consolidate(mesh)


def _rect(width, height, nx, ny):
    pts = [(width * i / nx, 0.0) for i in range(nx)]
    pts += [(width, height * i / ny) for i in range(ny)]
    pts += [(width - width * i / nx, height) for i in range(nx)]
    pts += [(0.0, height - height * i / ny) for i in range(ny)]
    return pts


def _circle(cx, cy, r, n, clockwise):
    out = []
    for i in range(n):
        t = i / n * math.tau
        if clockwise:
            t = -t
        out.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    return out


def _cells(mesh):
    """Cell count per element type."""
    return dict(zip(mesh.element_types(), mesh.cell_counts()))


def test_pave_surface_fills_a_square_with_quadrangles():
    coords = pc.Coords(2)
    contour = _closed_loop(coords, _rect(1.0, 1.0, 8, 8))
    mesh = pc.mesher.pave_surface(contour, "QUA4", size=0.125)
    cells = _cells(mesh)
    assert "QUA4" in cells
    assert cells.get("TRI3", 0) == 0
    assert mesh.cell_count() > 30


def test_pave_surface_handles_a_hole():
    coords = pc.Coords(2)
    outer = _closed_loop(coords, _rect(3.0, 1.0, 30, 10))
    hole = _closed_loop(coords, _circle(2.25, 0.5, 0.35, 32, clockwise=True))
    mesh = pc.mesher.pave_surface(outer | hole, "QUA4", size=0.1)
    assert mesh.cell_count() > 200
    # The hole is not meshed: the area is well short of the full rectangle.
    assert _cells(mesh).get("QUA4", 0) > 0


def test_pave_surface_all_quad_leaves_no_triangle():
    coords = pc.Coords(2)
    # An odd number of boundary segments, which is what forces a triangle.
    contour = _closed_loop(coords, _rect(1.0, 1.0, 4, 4) + [(0.0, 0.125)])
    mesh = pc.mesher.pave_surface(contour, "QUA4", size=0.25, all_quad=True)
    assert _cells(mesh).get("TRI3", 0) == 0


def test_pave_surface_default_size_follows_the_contour():
    coords = pc.Coords(2)
    contour = _closed_loop(coords, _rect(1.0, 1.0, 10, 10))
    mesh = pc.mesher.pave_surface(contour, "QUA4")
    assert 40 <= mesh.cell_count() <= 400


def test_pave_surface_accepts_quadratic_quadrangles():
    coords = pc.Coords(2)
    contour = _closed_loop(coords, _rect(1.0, 1.0, 4, 4))
    for name in ("QUA8", "QUA9"):
        mesh = pc.mesher.pave_surface(contour, name, size=0.25, all_quad=True)
        assert mesh.element_types() == [name]


def test_pave_surface_paves_a_planar_contour_in_3d():
    coords = pc.Coords(3)
    contour = _closed_loop(coords, [(x, 0.0, y) for x, y in _rect(1.0, 1.0, 6, 6)])
    mesh = pc.mesher.pave_surface(contour, "QUA4", size=0.2)
    assert mesh.cell_count() > 10


def test_pave_surface_extrudes_to_hexahedra():
    """The prismatic 3-D case: an all-quadrangle mesh extrudes to pure HEX8."""
    coords = pc.Coords(3)
    contour = _closed_loop(coords, [(x, 0.0, y) for x, y in _rect(1.0, 1.0, 6, 6)])
    plate = pc.mesher.pave_surface(contour, "QUA4", size=0.2, all_quad=True)
    assert _cells(plate).get("TRI3", 0) == 0
    volume = pc.mesher.extrude(plate, [0.0, 0.2, 0.0], 2)
    assert volume.element_types() == ["HEX8"]
    assert volume.cell_count() == 2 * plate.cell_count()


def test_pave_surface_rejects_a_non_quadrangle_element_type():
    coords = pc.Coords(2)
    contour = _closed_loop(coords, _rect(1.0, 1.0, 4, 4))
    with pytest.raises(Exception, match="pave_surface"):
        pc.mesher.pave_surface(contour, "TRI3", size=0.25)


def test_pave_surface_rejects_a_non_positive_size():
    coords = pc.Coords(2)
    contour = _closed_loop(coords, _rect(1.0, 1.0, 4, 4))
    with pytest.raises(Exception, match="pave_surface"):
        pc.mesher.pave_surface(contour, "QUA4", size=0.0)
