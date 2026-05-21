"""Python tests for NodeField (Phase 2 step 3)."""

import gc as pygc

import pyrucast


def _poi1_with(n_nodes, dim=2):
    c = pyrucast.Configuration(dim)
    coords = [0.0] * dim
    nodes = [c.add_node([float(i)] + coords[1:]) for i in range(n_nodes)]
    sm = pyrucast.SubMesh(c, "POI1")
    for n in nodes:
        sm.add_cell([n.id])
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
    f.set_by_node(nodes[1].id, ci_p, 42.0)
    assert f.get_by_node(nodes[1].id, ci_p) == 42.0


def test_unknown_node_or_component():
    c, _nodes, sm = _poi1_with(1)
    f = pyrucast.NodeField(sm, ["T"])
    assert f.component_index("missing") is None
    try:
        f.get_by_node(999, 0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for unknown NodeId")


def test_field_protects_nodes_from_gc():
    c = pyrucast.Configuration(1)
    a = c.add_node([0.0])
    nid = a.id
    sm = pyrucast.SubMesh(c, "POI1")
    sm.add_cell([nid])
    field = pyrucast.NodeField(sm, ["T"])

    # refcount = Node + SubMesh + NodeField = 3
    assert c.refcount(nid) == 3

    del a
    del sm
    pygc.collect()
    # Field still keeps it alive.
    assert c.refcount(nid) == 1
    assert c.is_alive(nid)
    assert c.gc() == 0

    del field
    pygc.collect()
    assert c.gc() == 1
    assert not c.is_alive(nid)


def test_repr_str_node_field():
    c, _nodes, sm = _poi1_with(3)
    f = pyrucast.NodeField(sm, ["UX", "UY"])
    assert "NodeField" in repr(f)
    s = str(f)
    assert "3 node(s)" in s
    assert "2 component(s)" in s
    assert "UX, UY" in s


