"""Python tests for SubMesh + Mesh (Phase 2 step 2)."""

import gc as pygc

import pyrucast


def test_submesh_poi1_is_node_list():
    c = pyrucast.Configuration(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    sm = pyrucast.Mesh(c, "POI1")[0]
    sm.add_cell([a])
    sm.add_cell([b])
    assert sm.cell_count() == 2
    assert sm.element_type == "POI1"


def test_poi1_from_nodes_builds_points_mesh():
    c = pyrucast.Configuration(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    # Node-based: the Configuration is taken from the nodes themselves.
    m = pyrucast.poi1_from_nodes([a, b])
    assert m.element_types() == ["POI1"]
    assert m.cell_count() == 2


def test_poi1_from_nodes_empty_raises():
    try:
        pyrucast.poi1_from_nodes([])
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for empty node list")


def test_aggregate_union_sub_and_sub_union_sub():
    c = pyrucast.Configuration(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    s1 = pyrucast.poi1_from_nodes([a])[0]  # a PySubMesh
    s2 = pyrucast.poi1_from_nodes([b])[0]

    # sub + sub → Mesh
    m = s1 | s2
    assert len(m) == 2

    # mesh + sub → Mesh
    s3 = pyrucast.poi1_from_nodes([a])[0]
    m2 = m | s3
    assert len(m2) == 3


def test_node_union_node_and_mesh_union_node():
    c = pyrucast.Configuration(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    d = c.add_node([2.0, 0.0])

    m = a | b  # node | node → unitary POI1 Mesh
    assert m.element_types() == ["POI1"]
    assert m.cell_count() == 2

    m2 = m | d  # mesh | node → unitary POI1 Mesh (via Node.__ror__)
    assert m2.cell_count() == 3


def test_mesh_union_node_rejects_non_poi1():
    c = pyrucast.Configuration(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    cc = c.add_node([0.5, 1.0])
    tri = pyrucast.Mesh(c, "TRI3")
    tri.unit().add_cell([a, b, cc])
    try:
        _ = tri | a  # not a unitary POI1 mesh
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for non-POI1 mesh + node")


def test_submesh_tri3_invalid_arity():
    c = pyrucast.Configuration(2)
    a = c.add_node([0.0, 0.0])
    sm = pyrucast.Mesh(c, "TRI3")[0]
    try:
        sm.add_cell([a])
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

    sm = pyrucast.Mesh(c, "TRI3")[0]
    sm.add_cell([a, b, cc])
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
        pyrucast.Mesh(c, "BOGUS")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown type")


def test_mesh_aggregates_submeshes():
    c = pyrucast.Configuration(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    cc = c.add_node([0.5, 1.0])

    pts = pyrucast.Mesh(c, "POI1")
    pts.unit().add_cell([a])
    pts.unit().add_cell([b])

    tri = pyrucast.Mesh(c, "TRI3")
    tri.unit().add_cell([a, b, cc])

    mesh = pts | tri
    assert len(mesh) == 2
    assert mesh.cell_count() == 3  # 2 POI1 + 1 TRI3


def test_submesh_cell_indexing_and_iteration():
    """`len(sm)` = cell count, `sm[i]` returns a Cell, and a Cell is
    itself iterable over its nodes."""
    c = pyrucast.Configuration(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    cc = c.add_node([0.5, 1.0])

    sm = pyrucast.Mesh(c, "TRI3")[0]
    sm.add_cell([a, b, cc])

    assert len(sm) == 1
    cell = sm[0]
    assert cell.element_type == "TRI3"
    assert cell.index == 0
    assert [n.id for n in cell.nodes()] == [a.id, b.id, cc.id]
    assert len(cell) == 3
    assert [n.id for n in cell] == [a.id, b.id, cc.id]
    assert cell[-1].id == cc.id

    # for-loop on the submesh yields one Cell per cell.
    assert [c.index for c in sm] == [0]

    try:
        _ = sm[5]
    except IndexError:
        pass
    else:
        raise AssertionError("expected IndexError on out-of-range cell")


def test_mesh_indexing_and_iteration():
    """`mesh[i]`, `len(mesh)`, `for sm in mesh:` should all work, and the
    `SubMesh` returned by `mesh[i]` shares storage with the parent mesh
    so colour changes survive."""
    c = pyrucast.Configuration(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    cc = c.add_node([0.5, 1.0])

    pts = pyrucast.Mesh(c, "POI1")
    pts.unit().add_cell([a])

    tri = pyrucast.Mesh(c, "TRI3")
    tri.unit().add_cell([a, b, cc])

    mesh = pts | tri

    assert len(mesh) == 2
    assert [sm.element_type for sm in mesh] == ["POI1", "TRI3"]
    assert mesh[-1].element_type == "TRI3"

    mesh[1].face_color = (1, 2, 3)
    assert mesh[1].face_color == (1, 2, 3)

    try:
        _ = mesh[5]
    except IndexError:
        pass
    else:
        raise AssertionError("expected IndexError on out-of-range index")


def test_mesh_rejects_merge_from_other_configuration():
    c1 = pyrucast.Configuration(2)
    c2 = pyrucast.Configuration(2)
    m1 = pyrucast.Mesh(c1, "POI1")
    m2 = pyrucast.Mesh(c2, "POI1")  # different config
    try:
        _ = m1 | m2  # merge across Configurations — must fail
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
        contour.unit().add_cell([nodes[i], nodes[(i + 1) % 4]])

    tri = pyrucast.fill_surface(contour, "TRI3")
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
        contour.unit().add_cell([nodes[i], nodes[(i + 1) % 3]])

    try:
        pyrucast.fill_surface(contour, "BOGUS")
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
        contour.unit().add_cell([nodes[i], nodes[(i + 1) % 3]])

    try:
        pyrucast.fill_surface(contour, "QUA4")
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
        contour.unit().add_cell([nodes[i], nodes[(i + 1) % n]])
    return contour, nodes


def test_fill_surface_with_one_hole_2d():
    # 4×4 outer square, 2×2 inner hole centred at (2, 2).
    c = pyrucast.Configuration(2)
    outer, _ = _build_seg2_loop(c, [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])
    hole, _ = _build_seg2_loop(c, [(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)])
    combined = outer | hole
    assert len(combined) == 2

    tri = pyrucast.fill_surface(combined, "TRI3")
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
    combined = hole | outer
    tri = pyrucast.fill_surface(combined, "TRI3")
    n_cells = tri.cell_count()

    total = 0.0
    for ci in range(n_cells):
        p0 = tri.node(0, ci, 0).coord()
        p1 = tri.node(0, ci, 1).coord()
        p2 = tri.node(0, ci, 2).coord()
        total += 0.5 * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]))
    assert abs(total - 12.0) < 1e-9


def test_fill_surface_refined_creates_more_triangles():
    # 4×4 square with max_edge_length=1.5 must produce strictly more
    # triangles than the un-refined 2.
    c = pyrucast.Configuration(2)
    outer, _ = _build_seg2_loop(c, [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])
    tri = pyrucast.fill_surface(outer, "TRI3", max_edge_length=1.5)
    assert tri.cell_count() > 2

    # Conservation of area: 4 × 4 = 16.
    total = 0.0
    for ci in range(tri.cell_count()):
        p0 = tri.node(0, ci, 0).coord()
        p1 = tri.node(0, ci, 1).coord()
        p2 = tri.node(0, ci, 2).coord()
        total += 0.5 * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]))
    assert abs(total - 16.0) < 1e-9


def test_fill_surface_refined_with_hole():
    # 4×4 square minus 2×2 hole, refined: area must still be 12.
    c = pyrucast.Configuration(2)
    outer, _ = _build_seg2_loop(c, [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])
    hole, _ = _build_seg2_loop(c, [(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)])
    combined = outer | hole
    tri = pyrucast.fill_surface(combined, "TRI3", max_edge_length=1.0)
    total = 0.0
    for ci in range(tri.cell_count()):
        p0 = tri.node(0, ci, 0).coord()
        p1 = tri.node(0, ci, 1).coord()
        p2 = tri.node(0, ci, 2).coord()
        total += 0.5 * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]))
    assert abs(total - 12.0) < 1e-9


def test_fill_surface_refined_angle_criterion():
    # 4×1 rectangle refined to min angle 20°: initial Delaunay has ~14°
    # somewhere; after refinement no triangle should be below ~19°.
    c = pyrucast.Configuration(2)
    rect, _ = _build_seg2_loop(c, [(0.0, 0.0), (4.0, 0.0), (4.0, 1.0), (0.0, 1.0)])
    tri = pyrucast.fill_surface(rect, "TRI3", min_angle_deg=20.0)
    # Compute min angle across all triangles.
    import math
    min_deg = math.inf
    for ci in range(tri.cell_count()):
        pts = [tri.node(0, ci, k).coord() for k in range(3)]
        for k in range(3):
            u, v, w = pts[k], pts[(k + 1) % 3], pts[(k + 2) % 3]
            e1 = (v[0] - u[0], v[1] - u[1])
            e2 = (w[0] - u[0], w[1] - u[1])
            cos = (e1[0] * e2[0] + e1[1] * e2[1]) / (
                math.sqrt(e1[0] ** 2 + e1[1] ** 2) * math.sqrt(e2[0] ** 2 + e2[1] ** 2)
            )
            cos = max(-1.0, min(1.0, cos))
            ang = math.degrees(math.acos(cos))
            min_deg = min(min_deg, ang)
    assert min_deg >= 19.0, f"min angle still bad: {min_deg} deg"


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
        contour.unit().add_cell([nodes[i], nodes[(i + 1) % 4]])

    tri = pyrucast.fill_surface(contour, "TRI3")
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
        contour.unit().add_cell([nodes[i], nodes[(i + 1) % 4]])

    try:
        pyrucast.fill_surface(contour, "TRI3")
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
    bogus.unit().add_cell([a, b, cc])

    try:
        pyrucast.fill_surface(bogus, "TRI3")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for non-SEG2 contour")


def test_to_poi1_converts_each_submesh_to_node_list():
    c = pyrucast.Configuration(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    cc = c.add_node([0.5, 1.0])
    d = c.add_node([1.5, 1.0])

    # Two triangles sharing edge (b, cc): 4 unique nodes.
    tri = pyrucast.Mesh(c, "TRI3")
    tri.unit().add_cell([a, b, cc])
    tri.unit().add_cell([b, d, cc])

    poi = pyrucast.to_poi1(tri)
    assert len(poi) == 1
    assert poi.element_types() == ["POI1"]
    assert poi.cell_count() == 4  # 6 connectivity entries, deduplicated
    ids = [poi.node(0, i, 0).id for i in range(4)]
    assert ids == [a.id, b.id, cc.id, d.id]


def test_to_poi1_preserves_submesh_count():
    c = pyrucast.Configuration(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    cc = c.add_node([0.5, 1.0])

    pts = pyrucast.Mesh(c, "POI1")
    pts.unit().add_cell([a])
    tri = pyrucast.Mesh(c, "TRI3")
    tri.unit().add_cell([a, b, cc])
    mesh = pts | tri

    poi = pyrucast.to_poi1(mesh)
    assert len(poi) == 2
    assert poi.element_types() == ["POI1", "POI1"]
    assert poi.cell_counts() == [1, 3]


def test_repr_str_submesh_and_mesh():
    c = pyrucast.Configuration(1)
    sm = pyrucast.Mesh(c, "SEG2")[0]
    assert "SubMesh" in repr(sm)
    assert "SEG2" in str(sm)
    mesh = pyrucast.Mesh(c)
    assert "Mesh" in repr(mesh)
    assert "submesh" in str(mesh)
