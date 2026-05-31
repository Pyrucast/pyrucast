"""Python tests for NodeField (Phase 2 step 3)."""

import gc as pygc

import pyrucast


def _poi1_with(n_nodes, dim=2):
    c = pyrucast.Configuration(dim)
    coords = [0.0] * dim
    nodes = [c.add_node([float(i)] + coords[1:]) for i in range(n_nodes)]
    sm = pyrucast.SubMesh(c, "POI1")
    for n in nodes:
        sm.add_cell([n])
    return c, nodes, sm


def test_from_poi1_zero_initialized():
    c, _nodes, sm = _poi1_with(3)
    f = pyrucast.NodeField(sm, ["T"])
    assert f.node_count() == 3
    assert f.component_count() == 1
    assert f.components() == ["T"]
    for i in range(3):
        assert f.get(i, 0) == 0.0


def test_from_poi1_rejects_non_poi1():
    c = pyrucast.Configuration(2)
    sm = pyrucast.SubMesh(c, "SEG2")
    try:
        pyrucast.NodeField(sm, ["X"])
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for non-POI1 SubMesh")


def test_from_poi1_rejects_empty_components():
    c, _nodes, sm = _poi1_with(1)
    try:
        pyrucast.NodeField(sm, [])
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for empty components")


def test_from_poi1_rejects_duplicate_components():
    c, _nodes, sm = _poi1_with(1)
    try:
        pyrucast.NodeField(sm, ["UX", "UX"])
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for duplicate components")


def test_get_set_multi_component():
    c, _nodes, sm = _poi1_with(2)
    f = pyrucast.NodeField(sm, ["UX", "UY", "UZ"])
    f.set(0, 0, 1.0)
    f.set(0, 1, 2.0)
    f.set(0, 2, 3.0)
    f.set(1, 1, -7.0)
    assert f.node_values(0) == [1.0, 2.0, 3.0]
    assert f.node_values(1) == [0.0, -7.0, 0.0]


def test_by_node_id_access():
    c, nodes, sm = _poi1_with(2)
    f = pyrucast.NodeField(sm, ["T", "P"])
    ci_p = f.component_index("P")
    assert ci_p == 1
    f.set_by_node(nodes[1], ci_p, 42.0)
    assert f.get_by_node(nodes[1], ci_p) == 42.0


def test_unknown_node_or_component():
    c, _nodes, sm = _poi1_with(1)
    f = pyrucast.NodeField(sm, ["T"])
    assert f.component_index("missing") is None
    other = c.add_node([99.0, 99.0])  # alive in the config but not in the field's support
    try:
        f.get_by_node(other, 0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for unknown NodeId")


def test_field_protects_nodes_from_gc():
    c = pyrucast.Configuration(1)
    a = c.add_node([0.0])
    nid = a.id
    sm = pyrucast.SubMesh(c, "POI1")
    sm.add_cell([a])
    field = pyrucast.NodeField(sm, ["T"])

    # NodeField shares the SubMesh handle, so per-node refcounts are
    # only Node + SubMesh = 2 (the field adds no per-node incref).
    assert c.refcount(nid) == 2

    del a
    del sm
    pygc.collect()
    # The field still holds a clone of the SubMesh handle, so the
    # SubMesh stays alive and keeps the node alive (refcount = 1).
    assert c.refcount(nid) == 1
    assert c.is_alive(nid)
    assert c.gc() == 0

    del field
    pygc.collect()
    assert c.gc() == 1
    assert not c.is_alive(nid)


def test_coordinates_poi1_mesh_xyz():
    c = pyrucast.Configuration(3)
    a = c.add_node([1.0, 2.0, 3.0])
    b = c.add_node([4.0, 5.0, 6.0])
    mesh = pyrucast.Mesh(c, "POI1")
    mesh.add_cell([a])
    mesh.add_cell([b])

    f = pyrucast.coordinates(mesh)
    assert f.components() == ["X", "Y", "Z"]
    assert f.node_count() == 2
    assert f[a, "X"] == 1.0
    assert f[a, "Y"] == 2.0
    assert f[a, "Z"] == 3.0
    assert f[b, "Z"] == 6.0


def test_coordinates_converts_non_poi1_and_deduplicates():
    c = pyrucast.Configuration(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    cc = c.add_node([0.5, 1.0])
    d = c.add_node([1.5, 1.0])

    tri = pyrucast.Mesh(c, "TRI3")
    tri.add_cell([a, b, cc])
    tri.add_cell([b, d, cc])

    f = pyrucast.coordinates(tri)
    assert f.components() == ["X", "Y"]  # 2-D ⇒ X, Y only
    assert f.node_count() == 4  # shared nodes appear once
    assert f[cc, "X"] == 0.5
    assert f[d, "X"] == 1.5


def test_coordinates_component_subset():
    c = pyrucast.Configuration(3)
    a = c.add_node([1.0, 2.0, 3.0])
    mesh = pyrucast.Mesh(c, "POI1")
    mesh.add_cell([a])

    f = pyrucast.coordinates(mesh, ["X", "Z"])
    assert f.components() == ["X", "Z"]
    assert f[a, "X"] == 1.0
    assert f[a, "Z"] == 3.0


def test_coordinates_rejects_axis_beyond_dimension():
    c = pyrucast.Configuration(2)
    a = c.add_node([0.0, 0.0])
    mesh = pyrucast.Mesh(c, "POI1")
    mesh.add_cell([a])
    try:
        pyrucast.coordinates(mesh, ["Z"])  # no Z in 2-D
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for Z on a 2-D mesh")


def test_repr_str_node_field():
    c, _nodes, sm = _poi1_with(3)
    f = pyrucast.NodeField(sm, ["UX", "UY"])
    assert "NodeField" in repr(f)
    s = str(f)
    assert "3 node(s)" in s
    assert "2 component(s)" in s
    assert "UX, UY" in s




# ─── Field arithmetic & combination (module operators) ──────────────────────


def test_add_scalar():
    c, nodes, sm = _poi1_with(2)
    f = pyrucast.NodeField(sm, ["T"])
    f.set_value(nodes[0], "T", 1.0)
    f.set_value(nodes[1], "T", 2.0)
    g = f + 10.0
    assert g.value(nodes[0], "T") == 11.0
    assert g.value(nodes[1], "T") == 12.0


def test_add_field_plus_field():
    # Regression: `a + b` previously deadlocked (nested per-type store lock).
    c, nodes, sm = _poi1_with(2)
    a = pyrucast.NodeField(sm, ["T"])
    b = pyrucast.NodeField(sm, ["T"])
    a.set_value(nodes[0], "T", 1.0)
    b.set_value(nodes[0], "T", 10.0)
    b.set_value(nodes[1], "T", 5.0)
    c2 = a + b
    assert c2.value(nodes[0], "T") == 11.0
    assert c2.value(nodes[1], "T") == 5.0


def test_add_field_to_itself_does_not_deadlock():
    # Same handle on both sides must not re-lock the store mutex.
    c, nodes, sm = _poi1_with(1)
    a = pyrucast.NodeField(sm, ["T"])
    a.set_value(nodes[0], "T", 3.0)
    assert (a + a).value(nodes[0], "T") == 6.0


def test_add_rejects_bad_operand():
    c, _nodes, sm = _poi1_with(1)
    f = pyrucast.NodeField(sm, ["T"])
    try:
        f + "nope"
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for str operand")


def test_merge_compatible_and_conflict():
    # a on {n0, n1}, b on {n1, n2} — n1 shared. Union = {n0, n1, n2}.
    c = pyrucast.Configuration(1)
    n0 = c.add_node([0.0])
    n1 = c.add_node([1.0])
    n2 = c.add_node([2.0])
    sm_a = pyrucast.SubMesh(c, "POI1")
    sm_a.add_cell([n0])
    sm_a.add_cell([n1])
    sm_b = pyrucast.SubMesh(c, "POI1")
    sm_b.add_cell([n1])
    sm_b.add_cell([n2])
    a = pyrucast.NodeField(sm_a, ["T"])
    b = pyrucast.NodeField(sm_b, ["T"])
    a.set_value(n0, "T", 5.0)
    a.set_value(n1, "T", 3.0)
    b.set_value(n1, "T", 3.0)  # same value at the shared node → compatible
    b.set_value(n2, "T", 9.0)

    m = pyrucast.merge(a, b)
    assert m.node_count() == 3
    assert m.value(n0, "T") == 5.0
    assert m.value(n1, "T") == 3.0
    assert m.value(n2, "T") == 9.0

    # Conflicting value at the shared node → error.
    b.set_value(n1, "T", 7.0)
    try:
        pyrucast.merge(a, b)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError on conflicting merge")


def test_restrict_to_mesh_subset():
    c, nodes, sm = _poi1_with(3)
    f = pyrucast.NodeField(sm, ["T"])
    for i, n in enumerate(nodes):
        f.set_value(n, "T", float(i + 1))

    # Mesh covering only nodes[0] and nodes[2].
    mesh = pyrucast.Mesh(c, "POI1")
    mesh.add_cell([nodes[0]])
    mesh.add_cell([nodes[2]])

    r = pyrucast.restrict(f, mesh)
    assert r.node_count() == 2
    assert r.value(nodes[0], "T") == 1.0
    assert r.value(nodes[2], "T") == 3.0
