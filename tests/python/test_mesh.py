"""Python tests for SubMesh + Mesh (Phase 2 step 2)."""

import gc as pygc

import pyrucast


def test_submesh_poi1_is_node_list():
    c = pyrucast.Configuration(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    sm = pyrucast.SubMesh(c, "POI1")
    sm.add_cell([a.id])
    sm.add_cell([b.id])
    assert sm.cell_count() == 2
    assert sm.element_type == "POI1"


def test_submesh_tri3_invalid_arity():
    c = pyrucast.Configuration(2)
    a = c.add_node([0.0, 0.0])
    sm = pyrucast.SubMesh(c, "TRI3")
    try:
        sm.add_cell([a.id])
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for invalid arity")


def test_submesh_protects_nodes_from_gc():
    c = pyrucast.Configuration(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    cc = c.add_node([0.5, 1.0])
    ids = (a.id, b.id, cc.id)

    sm = pyrucast.SubMesh(c, "TRI3")
    sm.add_cell(list(ids))
    # The 3 Python Nodes AND the SubMesh each reference every node ⇒ refcount = 2.
    for i in ids:
        assert c.refcount(i) == 2

    # Drop the Python Nodes; the SubMesh keeps the nodes alive.
    del a
    del b
    del cc
    pygc.collect()
    for i in ids:
        assert c.refcount(i) == 1
        assert c.is_alive(i)
    assert c.gc() == 0

    # Drop the SubMesh ⇒ refcount drops to 0 ⇒ gc collects.
    del sm
    pygc.collect()
    assert c.gc() == 3
    for i in ids:
        assert not c.is_alive(i)


def test_unknown_element_type():
    c = pyrucast.Configuration(2)
    try:
        pyrucast.SubMesh(c, "BOGUS")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown type")


def test_mesh_aggregates_submeshes():
    c = pyrucast.Configuration(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    cc = c.add_node([0.5, 1.0])

    sm_pts = pyrucast.SubMesh(c, "POI1")
    sm_pts.add_cell([a.id])
    sm_pts.add_cell([b.id])

    sm_tri = pyrucast.SubMesh(c, "TRI3")
    sm_tri.add_cell([a.id, b.id, cc.id])

    mesh = pyrucast.Mesh(c)
    mesh.add_submesh(sm_pts)
    mesh.add_submesh(sm_tri)
    assert mesh.submesh_count() == 2
    assert mesh.cell_count() == 3  # 2 POI1 + 1 TRI3


def test_mesh_rejects_submesh_from_other_configuration():
    c1 = pyrucast.Configuration(2)
    c2 = pyrucast.Configuration(2)
    sm = pyrucast.SubMesh(c1, "POI1")
    mesh = pyrucast.Mesh(c2)
    try:
        mesh.add_submesh(sm)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for mismatched Configurations")


def test_fill_surface_square_gives_two_triangles():
    c = pyrucast.Configuration(2)
    nodes = [
        c.add_node([0.0, 0.0]),
        c.add_node([1.0, 0.0]),
        c.add_node([1.0, 1.0]),
        c.add_node([0.0, 1.0]),
    ]
    contour = pyrucast.Mesh(c, "SEG2")
    for i in range(4):
        contour.add_cell([nodes[i].id, nodes[(i + 1) % 4].id])

    tri = pyrucast.Mesh.fill_surface(contour, "TRI3")
    assert tri.element_types() == ["TRI3"]
    assert tri.cell_count() == 2  # n - 2


def test_fill_surface_unknown_element_type():
    c = pyrucast.Configuration(2)
    nodes = [
        c.add_node([0.0, 0.0]),
        c.add_node([1.0, 0.0]),
        c.add_node([0.5, 1.0]),
    ]
    contour = pyrucast.Mesh(c, "SEG2")
    for i in range(3):
        contour.add_cell([nodes[i].id, nodes[(i + 1) % 3].id])

    try:
        pyrucast.Mesh.fill_surface(contour, "BOGUS")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown element type")


def test_fill_surface_rejects_unsupported_target_element():
    c = pyrucast.Configuration(2)
    nodes = [
        c.add_node([0.0, 0.0]),
        c.add_node([1.0, 0.0]),
        c.add_node([0.5, 1.0]),
    ]
    contour = pyrucast.Mesh(c, "SEG2")
    for i in range(3):
        contour.add_cell([nodes[i].id, nodes[(i + 1) % 3].id])

    try:
        pyrucast.Mesh.fill_surface(contour, "QUA4")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for unsupported target type")


def _build_seg2_loop(c, pts):
    """Helper: create a closed SEG2 Mesh from a list of (x, y[, z]) points."""
    nodes = [c.add_node(list(p)) for p in pts]
    contour = pyrucast.Mesh(c, "SEG2")
    n = len(nodes)
    for i in range(n):
        contour.add_cell([nodes[i].id, nodes[(i + 1) % n].id])
    return contour, nodes


def test_fill_surface_with_one_hole_2d():
    # 4×4 outer square, 2×2 inner hole centred at (2, 2).
    c = pyrucast.Configuration(2)
    outer, _ = _build_seg2_loop(c, [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])
    hole, _ = _build_seg2_loop(c, [(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)])
    combined = outer + hole
    assert combined.submesh_count() == 2

    tri = pyrucast.Mesh.fill_surface(combined, "TRI3")
    n_cells = tri.cell_count()

    # Triangulated area should equal outer 16 minus hole 4 = 12.
    total = 0.0
    for ci in range(n_cells):
        p0 = tri.node(0, ci, 0).coord()
        p1 = tri.node(0, ci, 1).coord()
        p2 = tri.node(0, ci, 2).coord()
        total += 0.5 * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]))
    assert abs(total - 12.0) < 1e-9


def test_fill_surface_outer_loop_autodetected():
    # Pass the hole first; the outer loop is still detected correctly.
    c = pyrucast.Configuration(2)
    hole, _ = _build_seg2_loop(c, [(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)])
    outer, _ = _build_seg2_loop(c, [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])
    combined = hole + outer
    tri = pyrucast.Mesh.fill_surface(combined, "TRI3")
    n_cells = tri.cell_count()

    total = 0.0
    for ci in range(n_cells):
        p0 = tri.node(0, ci, 0).coord()
        p1 = tri.node(0, ci, 1).coord()
        p2 = tri.node(0, ci, 2).coord()
        total += 0.5 * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]))
    assert abs(total - 12.0) < 1e-9


def test_fill_surface_3d_tilted_square():
    # Square rotated 45° around the x axis: planar in 3-D, must triangulate.
    import math

    s = 1.0 / math.sqrt(2.0)
    c = pyrucast.Configuration(3)
    pts = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, s, s),
        (0.0, s, s),
    ]
    nodes = [c.add_node(list(p)) for p in pts]
    contour = pyrucast.Mesh(c, "SEG2")
    for i in range(4):
        contour.add_cell([nodes[i].id, nodes[(i + 1) % 4].id])

    tri = pyrucast.Mesh.fill_surface(contour, "TRI3")
    assert tri.cell_count() == 2  # n - 2


def test_fill_surface_3d_rejects_non_planar_contour():
    c = pyrucast.Configuration(3)
    # One corner well out of the z = 0 plane.
    pts = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.5),
        (0.0, 1.0, 0.0),
    ]
    nodes = [c.add_node(list(p)) for p in pts]
    contour = pyrucast.Mesh(c, "SEG2")
    for i in range(4):
        contour.add_cell([nodes[i].id, nodes[(i + 1) % 4].id])

    try:
        pyrucast.Mesh.fill_surface(contour, "TRI3")
    except RuntimeError as e:
        assert "not planar" in str(e)
    else:
        raise AssertionError("expected RuntimeError for non-planar 3-D contour")


def test_fill_surface_rejects_non_seg2_contour():
    c = pyrucast.Configuration(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    cc = c.add_node([0.5, 1.0])
    bogus = pyrucast.Mesh(c, "TRI3")
    bogus.add_cell([a.id, b.id, cc.id])

    try:
        pyrucast.Mesh.fill_surface(bogus, "TRI3")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for non-SEG2 contour")


def test_repr_str_submesh_and_mesh():
    c = pyrucast.Configuration(1)
    sm = pyrucast.SubMesh(c, "SEG2")
    assert "SubMesh" in repr(sm)
    assert "SEG2" in str(sm)
    mesh = pyrucast.Mesh(c)
    assert "Mesh" in repr(mesh)
    assert "submesh" in str(mesh)
