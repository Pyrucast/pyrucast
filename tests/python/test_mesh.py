"""Python tests for SubMesh + Mesh (Phase 2 step 2)."""

import gc as pygc

import pyrucast


def test_submesh_poi1_is_node_list():
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    sm = pyrucast.Mesh(c, "POI1")[0]
    sm.add_cell([a])
    sm.add_cell([b])
    assert sm.cell_count() == 2
    assert sm.element_type == "POI1"


def test_poi1_from_nodes_builds_points_mesh():
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    # Node-based: the Coords is taken from the nodes themselves.
    m = pyrucast.Mesh.poi1_from_nodes([a, b])
    assert m.element_types() == ["POI1"]
    assert m.cell_count() == 2


def test_poi1_from_nodes_empty_raises():
    try:
        pyrucast.Mesh.poi1_from_nodes([])
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for empty node list")


def test_submesh_sealed_by_fe_space_refuses_add_cell():
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    d = c.add_node([0.0, 1.0])
    e = c.add_node([1.0, 1.0])
    mesh = pyrucast.Mesh(c, "TRI3")
    sm = mesh[0]
    sm.add_cell([a, b, d])
    assert sm.is_sealed is False
    # Using the mesh in a finite-element space seals its submesh.
    pyrucast.FiniteElementSpace(mesh)
    assert sm.is_sealed is True
    try:
        sm.add_cell([b, d, e])
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError adding a cell to a sealed submesh")
    assert sm.cell_count() == 1


def test_duplicate_gives_editable_copy_after_seal():
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    d = c.add_node([0.0, 1.0])
    e = c.add_node([1.0, 1.0])
    mesh = pyrucast.Mesh(c, "TRI3")
    mesh[0].add_cell([a, b, d])
    pyrucast.FiniteElementSpace(mesh)  # seals mesh[0]

    copy = mesh.duplicate()
    assert copy.cell_count() == 1
    assert copy[0].is_sealed is False
    # The copy can keep growing while the original stays frozen.
    copy[0].add_cell([b, d, e])
    assert copy.cell_count() == 2
    assert mesh.cell_count() == 1


def test_aggregate_union_sub_and_sub_union_sub():
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    s1 = pyrucast.Mesh.poi1_from_nodes([a])[0]  # a PySubMesh
    s2 = pyrucast.Mesh.poi1_from_nodes([b])[0]

    # sub + sub → Mesh
    m = s1 | s2
    assert len(m) == 2

    # mesh + sub → Mesh
    s3 = pyrucast.Mesh.poi1_from_nodes([a])[0]
    m2 = m | s3
    assert len(m2) == 3


def test_sub_union_aggregate_mirrors_aggregate_union_sub():
    """`sub | agg` (Mesh.__ror__) is accepted like `agg | sub`, sub first."""
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    d = c.add_node([2.0, 0.0])
    two = pyrucast.mesh.line(a, b, 2)  # one submesh, 2 cells
    five = pyrucast.mesh.line(b, d, 5)  # one submesh, 5 cells

    assert (two | five[0]).cell_counts() == [2, 5]
    assert (five[0] | two).cell_counts() == [5, 2]

    # A zone already held is not duplicated, it only moves to the front.
    m = two | five
    assert (m[1] | m).cell_counts() == [5, 2]

    # An unrelated left operand still raises, rather than silently unioning.
    try:
        _ = 3 | m
    except TypeError:
        pass
    else:
        raise AssertionError("int | Mesh should raise TypeError")


def test_node_union_node_and_mesh_union_node():
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    d = c.add_node([2.0, 0.0])

    m = a | b  # node | node → unitary POI1 Mesh
    assert m.element_types() == ["POI1"]
    assert m.cell_count() == 2

    m2 = m | d  # mesh | node → unitary POI1 Mesh (via Node.__ror__)
    assert m2.cell_count() == 3


def test_mesh_union_node_rejects_non_poi1():
    c = pyrucast.Coords(2)
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
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    sm = pyrucast.Mesh(c, "TRI3")[0]
    try:
        sm.add_cell([a])
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for invalid arity")


def test_submesh_protects_nodes_from_gc():
    c = pyrucast.Coords(2)
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
    c = pyrucast.Coords(2)
    try:
        pyrucast.Mesh(c, "BOGUS")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown type")


def test_mesh_aggregates_submeshes():
    c = pyrucast.Coords(2)
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
    c = pyrucast.Coords(2)
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
    c = pyrucast.Coords(2)
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


def test_mesh_rejects_merge_from_other_coords():
    c1 = pyrucast.Coords(2)
    c2 = pyrucast.Coords(2)
    m1 = pyrucast.Mesh(c1, "POI1")
    m2 = pyrucast.Mesh(c2, "POI1")  # different config
    try:
        _ = m1 | m2  # merge across Coords — must fail
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for mismatched Coords")


def _triangle_area_sum(tri):
    """Signed-area sum over the first (TRI3) submesh."""
    total = 0.0
    for ci in range(tri.cell_counts()[0]):
        p0 = tri.node(0, ci, 0).position()
        p1 = tri.node(0, ci, 1).position()
        p2 = tri.node(0, ci, 2).position()
        total += 0.5 * (
            (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        )
    return total


def _discretized_square(c, side=1.0, per_side=8):
    # Square contour with `per_side` SEG2 per edge. `triangulate_surface`
    # freezes the contour (never subdivides it), so a mesh finer than the input
    # boundary requires a pre-discretized contour — as real usage does.
    corners = [(0.0, 0.0), (side, 0.0), (side, side), (0.0, side)]
    ids = []
    for k in range(4):
        x0, y0 = corners[k]
        x1, y1 = corners[(k + 1) % 4]
        for i in range(per_side):
            t = i / per_side
            ids.append(c.add_node([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t]))
    contour = pyrucast.Mesh(c, "SEG2")
    n = len(ids)
    for i in range(n):
        contour.unit().add_cell([ids[i], ids[(i + 1) % n]])
    return contour


def test_triangulate_surface_square_tri3():
    # A unit square meshed with interior nodes: several triangles, area 1.
    c = pyrucast.Coords(2)
    contour = _discretized_square(c)

    tri = pyrucast.mesh.triangulate_surface(contour, "TRI3", size=0.25)
    assert tri.element_types() == ["TRI3"]
    assert tri.cell_count() > 2  # interior nodes created
    assert abs(_triangle_area_sum(tri) - 1.0) < 1e-9


def test_triangulate_surface_square_qua4_is_quad_dominant():
    c = pyrucast.Coords(2)
    contour = _discretized_square(c, per_side=6)

    quad = pyrucast.mesh.triangulate_surface(contour, "QUA4", size=0.25)
    # Quad-dominant: at least a QUA4 submesh (a few boundary TRI3 may remain).
    assert "QUA4" in quad.element_types()


def test_triangulate_surface_freezes_contour():
    # The result reuses exactly the input contour nodes (same id + position)
    # and adds no node on a contour edge.
    c = pyrucast.Coords(2)
    contour = _discretized_square(c, per_side=10)

    def node_map(mesh):
        s = {}
        for si, cnt in enumerate(mesh.cell_counts()):
            for ci in range(cnt):
                for nd in mesh.cell(si, ci).nodes():
                    s[nd.id] = tuple(nd.position())
        return s

    before = node_map(contour)
    mesh = pyrucast.mesh.triangulate_surface(contour, "TRI3", size=0.05)
    after = node_map(mesh)

    for nid, pos in before.items():
        assert nid in after, f"contour node {nid} dropped"
        assert after[nid] == pos, f"contour node {nid} moved"
    # No new node lies on an axis-aligned contour edge.
    for nid, (x, y) in after.items():
        if nid in before:
            continue
        on_edge = min(x, 1.0 - x, y, 1.0 - y) <= 1e-12
        assert not on_edge, f"new node {nid} placed on a contour edge at ({x}, {y})"


def test_triangulate_surface_unknown_element_type():
    c = pyrucast.Coords(2)
    nodes = [
        c.add_node([0.0, 0.0]),
        c.add_node([1.0, 0.0]),
        c.add_node([0.5, 1.0]),
    ]
    contour = pyrucast.Mesh(c, "SEG2")
    for i in range(3):
        contour.unit().add_cell([nodes[i], nodes[(i + 1) % 3]])

    try:
        pyrucast.mesh.triangulate_surface(contour, "BOGUS")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown element type")


def test_triangulate_surface_rejects_unsupported_target_element():
    c = pyrucast.Coords(2)
    nodes = [
        c.add_node([0.0, 0.0]),
        c.add_node([1.0, 0.0]),
        c.add_node([0.5, 1.0]),
    ]
    contour = pyrucast.Mesh(c, "SEG2")
    for i in range(3):
        contour.unit().add_cell([nodes[i], nodes[(i + 1) % 3]])

    try:
        pyrucast.mesh.triangulate_surface(contour, "TET4")
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


def test_triangulate_surface_with_one_hole_2d():
    # 4×4 outer square (CCW), 2×2 inner hole centred at (2, 2), given CW.
    c = pyrucast.Coords(2)
    outer, _ = _build_seg2_loop(c, [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])
    hole, _ = _build_seg2_loop(c, [(1.0, 1.0), (1.0, 3.0), (3.0, 3.0), (3.0, 1.0)])
    combined = outer | hole
    assert len(combined) == 2

    tri = pyrucast.mesh.triangulate_surface(combined, "TRI3", size=1.0)
    # Meshed area = outer 16 minus hole 4 = 12.
    assert abs(_triangle_area_sum(tri) - 12.0) < 1e-9


def test_triangulate_surface_two_disjoint_domains():
    # Two disjoint CCW squares meshed in one pass.
    c = pyrucast.Coords(2)
    a, _ = _build_seg2_loop(c, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    b, _ = _build_seg2_loop(c, [(3.0, 0.0), (4.0, 0.0), (4.0, 1.0), (3.0, 1.0)])
    combined = a | b
    tri = pyrucast.mesh.triangulate_surface(combined, "TRI3", size=0.5)
    # Total meshed area = 1 + 1 = 2.
    assert abs(_triangle_area_sum(tri) - 2.0) < 1e-9


def test_triangulate_surface_rejects_all_holes_no_outer():
    # A single CW loop is a hole with no containing outer loop: error.
    c = pyrucast.Coords(2)
    hole, _ = _build_seg2_loop(c, [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)])
    try:
        pyrucast.mesh.triangulate_surface(hole, "TRI3")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError when no outer loop is given")


def test_triangulate_surface_3d_tilted_square():
    # Square rotated 45° around the x axis: planar in 3-D, must mesh.
    import math

    s = 1.0 / math.sqrt(2.0)
    c = pyrucast.Coords(3)
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

    tri = pyrucast.mesh.triangulate_surface(contour, "TRI3", size=0.5)
    assert tri.element_types() == ["TRI3"]
    assert tri.cell_count() >= 2


def test_triangulate_surface_3d_rejects_degenerate_contour():
    # A collinear 3-D "loop" has no well-defined plane and must be rejected.
    c = pyrucast.Coords(3)
    pts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)]
    nodes = [c.add_node(list(p)) for p in pts]
    contour = pyrucast.Mesh(c, "SEG2")
    for i in range(4):
        contour.unit().add_cell([nodes[i], nodes[(i + 1) % 4]])

    try:
        pyrucast.mesh.triangulate_surface(contour, "TRI3")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for a degenerate contour")


def test_triangulate_surface_rejects_non_seg2_contour():
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    cc = c.add_node([0.5, 1.0])
    bogus = pyrucast.Mesh(c, "TRI3")
    bogus.unit().add_cell([a, b, cc])

    try:
        pyrucast.mesh.triangulate_surface(bogus, "TRI3")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for non-SEG2 contour")


def test_to_poi1_converts_each_submesh_to_node_list():
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    cc = c.add_node([0.5, 1.0])
    d = c.add_node([1.5, 1.0])

    # Two triangles sharing edge (b, cc): 4 unique nodes.
    tri = pyrucast.Mesh(c, "TRI3")
    tri.unit().add_cell([a, b, cc])
    tri.unit().add_cell([b, d, cc])

    poi = pyrucast.mesh.to_poi1(tri)
    assert len(poi) == 1
    assert poi.element_types() == ["POI1"]
    assert poi.cell_count() == 4  # 6 connectivity entries, deduplicated
    ids = [poi.node(0, i, 0).id for i in range(4)]
    assert ids == [a.id, b.id, cc.id, d.id]


def test_to_poi1_preserves_submesh_count():
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    cc = c.add_node([0.5, 1.0])

    pts = pyrucast.Mesh(c, "POI1")
    pts.unit().add_cell([a])
    tri = pyrucast.Mesh(c, "TRI3")
    tri.unit().add_cell([a, b, cc])
    mesh = pts | tri

    poi = pyrucast.mesh.to_poi1(mesh)
    assert len(poi) == 2
    assert poi.element_types() == ["POI1", "POI1"]
    assert poi.cell_counts() == [1, 3]


def test_repr_str_submesh_and_mesh():
    c = pyrucast.Coords(1)
    sm = pyrucast.Mesh(c, "SEG2")[0]
    assert "SubMesh" in repr(sm)
    assert "SEG2" in str(sm)
    mesh = pyrucast.Mesh(c)
    assert "Mesh" in repr(mesh)
    assert "submesh" in str(mesh)


def _mesh_with_n_zones(n):
    """A Mesh of `n` single-node POI1 zones sharing the same Coords."""
    c = pyrucast.Coords(2)
    subs = [
        pyrucast.Mesh.poi1_from_nodes([c.add_node([float(i), 0.0])])[0]
        for i in range(n)
    ]
    mesh = subs[0]
    for s in subs[1:]:
        mesh = mesh | s
    return mesh


def test_slice_returns_aggregate_of_same_type():
    mesh = _mesh_with_n_zones(4)
    s = mesh[1:3]
    assert type(s) is pyrucast.Mesh
    assert len(s) == 2


def test_slice_full_step_and_open_bounds():
    mesh = _mesh_with_n_zones(4)
    assert len(mesh[:]) == 4  # full copy
    assert len(mesh[::2]) == 2  # one out of two
    assert len(mesh[1:]) == 3  # all but first
    assert len(mesh[:2]) == 2  # first two


def test_slice_negative_bounds_and_step():
    mesh = _mesh_with_n_zones(4)
    assert len(mesh[-2:]) == 2  # last two
    assert len(mesh[::-1]) == 4  # reversed
    # reversed slice preserves the zones, order flipped
    rev = mesh[::-1]
    fwd = mesh[:]
    assert rev[0].cell_count() == fwd[-1].cell_count()


def test_slice_empty():
    mesh = _mesh_with_n_zones(4)
    assert len(mesh[2:2]) == 0
    assert len(mesh[10:20]) == 0


def test_integer_index_still_returns_view():
    mesh = _mesh_with_n_zones(3)
    assert "SubMesh" in repr(mesh[0])
    assert "SubMesh" in repr(mesh[-1])


def test_index_errors():
    mesh = _mesh_with_n_zones(3)
    try:
        mesh[99]
    except IndexError:
        pass
    else:
        raise AssertionError("expected IndexError for out-of-range index")
    try:
        mesh["x"]
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for non-int/slice key")


def _hex_cube():
    """Unit HEX8 cube (bottom CCW then top CCW)."""
    c = pyrucast.Coords(3)
    pts = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]
    ids = [c.add_node(p) for p in pts]
    m = pyrucast.Mesh(c, "HEX8")
    m[0].add_cell(ids)
    return m


def test_skin_hex_cube_gives_six_quad_faces():
    sk = pyrucast.mesh.skin(_hex_cube())
    assert sk.element_types() == ["QUA4"] * 6
    assert sk.cell_count() == 6


def test_skin_extruded_solid_merges_flat_faces():
    # A triangulated square extruded into a PENTA6 box: skin must recover the
    # six flat faces — two triangulated caps and four quadrilateral walls —
    # regardless of how many cells tile each side.
    c = pyrucast.Coords(3)
    sq = [c.add_node(p) for p in [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]]
    contour = pyrucast.Mesh(c, "SEG2")
    for i in range(4):
        contour[0].add_cell([sq[i], sq[(i + 1) % 4]])
    surf = pyrucast.mesh.triangulate_surface(contour, "TRI3", size=0.34)
    solid = pyrucast.mesh.extrude(surf, [0, 0, 1], 3)

    faces = list(pyrucast.mesh.skin(solid))
    assert len(faces) == 6
    n_tri = sum(1 for s in faces if s.element_type == "TRI3")
    n_quad = sum(1 for s in faces if s.element_type == "QUA4")
    assert (n_tri, n_quad) == (2, 4)


def test_skin_large_angle_merges_all_faces():
    sk = pyrucast.mesh.skin(_hex_cube(), angle_deg=180.0)
    assert sk.element_types() == ["QUA4"]
    assert sk.cell_count() == 6


def test_skin_rejects_surface_mesh():
    c = pyrucast.Coords(3)
    a = c.add_node([0, 0, 0])
    b = c.add_node([1, 0, 0])
    d = c.add_node([0, 1, 0])
    m = pyrucast.Mesh(c, "TRI3")
    m[0].add_cell([a, b, d])
    try:
        pyrucast.mesh.skin(m)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError skinning a surface mesh")


def _unit_quad():
    """Unit QUA4 (CCW)."""
    c = pyrucast.Coords(2)
    ids = [c.add_node(p) for p in ([0, 0], [1, 0], [1, 1], [0, 1])]
    m = pyrucast.Mesh(c, "QUA4")
    m[0].add_cell(ids)
    return m


def test_convert_qua4_to_tri3_splits_each_quad():
    tri = pyrucast.mesh.convert(_unit_quad(), "TRI3")
    assert tri.element_types() == ["TRI3"]
    assert tri.cell_count() == 2


def test_convert_hex8_to_tet4_gives_six_tets():
    tets = pyrucast.mesh.convert(_hex_cube(), "TET4")
    assert tets.element_types() == ["TET4"]
    assert tets.cell_count() == 6


def test_convert_identity_is_noop_copy():
    tri = pyrucast.mesh.convert(_unit_quad(), "TRI3")
    same = pyrucast.mesh.convert(tri, "TRI3")
    assert same.element_types() == ["TRI3"]
    assert same.cell_count() == 2


def test_convert_rejects_unsupported_pair():
    tri = pyrucast.mesh.convert(_unit_quad(), "TRI3")
    try:
        pyrucast.mesh.convert(tri, "QUA4")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError converting TRI3 -> QUA4")


def test_convert_unknown_element_type():
    try:
        pyrucast.mesh.convert(_unit_quad(), "NOPE")
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError("expected error for unknown element type")


def test_consolidate_mesh_fuses_same_type_and_drops_duplicates():
    # Two submeshes of one element type carrying the same cell: consolidate_mesh
    # fuses them into a single submesh and drops the duplicate.
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    d = c.add_node([0.0, 1.0])
    za = pyrucast.Mesh(c, "TRI3")
    za.unit().add_cell([a, b, d])
    zb = pyrucast.Mesh(c, "TRI3")
    zb.unit().add_cell([a, b, d])  # duplicate cell in a second submesh
    m = pyrucast.mesh.consolidate(za | zb)
    assert len(m) == 1  # one submesh per element type, duplicates dropped
    assert m.cell_counts() == [1]
